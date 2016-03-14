#include "harens-re.h"
using namespace std;

HarensRE::HarensRE(int mapperNum, int reducerNum) 
{
	this->mapperNum = mapperNum;
	this->reducerNum = reducerNum;
	segmentThreads = new thread[mapperNum];
	tChunkMatch = new thread[reducerNum];
	circHashPool = new LRUStrHash<SHA1_HASH_LENGTH>[reducerNum];
	circHashPoolMutex = new mutex[reducerNum];
	duplicationSize = new unsigned long long[reducerNum];
	for (int i = 0; i < reducerNum; ++i) 
	{
		circHashPool[i] = LRUStrHash<SHA1_HASH_LENGTH>(MAX_CHUNK_NUM / reducerNum);
		duplicationSize[i] = 0;
	}

	re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::NonMultifingerprint);
	// initialize pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) 
	{
		pagableBuffer[i] = new char[MAX_BUFFER_LEN];
		pagableBufferObsolete[i] = true;
	}
	// initialize fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaMallocHost((void**)&fixedBuffer[i], MAX_BUFFER_LEN);
		// fixed_buffer_obsolete[i] = true;
	}
	// initialize chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaMalloc((void**)&kernelInputBuffer[i], MAX_BUFFER_LEN);
		cudaMalloc((void**)&kernelResultBuffer[i], MAX_WINDOW_NUM * BYTES_IN_UINT);
		cudaMallocHost((void**)&hostResultBuffer[i], MAX_WINDOW_NUM * BYTES_IN_UINT);
		hostResultObsolete[i] = true;
		hostResultExecuting[i] = false;
	}
	// initialize chunking result processing
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) 
	{
		cudaStreamCreate(&stream[i]);
		chunkingResultBuffer[i] = new unsigned int[MAX_WINDOW_NUM];
		chunkingResultObsolete[i] = true;
	}
}

HarensRE::~HarensRE() 
{
	delete[] segmentThreads;
	delete[] tChunkMatch;
	delete[] circHashPool;
	delete[] circHashPoolMutex;
	delete[] duplicationSize;

	// destruct chunking result proc
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) 
	{
		cudaStreamDestroy(stream[i]);
		delete[] chunkingResultBuffer[i];
	}
	// destruct chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaFree(kernelInputBuffer[i]);
		cudaFree(kernelResultBuffer[i]);
		cudaFreeHost(hostResultBuffer[i]);
	}
	// destruct fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaFreeHost(fixedBuffer[i]);
	}
	// destruct pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) 
	{
		delete[] pagableBuffer[i];
	}
}

void
HarensRE::HandleGetRequest(string request,
						   vector< tuple<int, unsigned char*, int, char*> >* result,
						   int& resultLenInUint8)
{
	// put values into request list as reference
	mutex resultMutex;
	condition_variable resultCond;
	result = new vector< tuple<int, unsigned char*, int, char*> >();
	resultLenInUint8 = 0;
	unique_lock<mutex> requestQueueLock(requestQueueMutex);
	requestQueue.push(make_tuple(ref(request), 
								 ref(result), 
								 ref(resultLenInUint8),
								 ref(resultMutex),
								 ref(resultCond)));
	requestQueueLock.unlock();
	newRequestCond.notify_all();
	// wait for result notification
	unique_lock<mutex> resultLock(resultMutex);
	resultCond.wait(resultLock, [result]{return result->size() > 0;});
}

void
HarensRE::Start()
{
	IO::Print("redundancy elimination module kernel started...\n");
	// initiate and start threads
	tReadData = thread(std::mem_fn(&HarensRE::ReadData), this);
	tChunkingKernel = thread(std::mem_fn(&HarensRE::ChunkingKernel), this);
	tChunkingResultProc = thread(std::mem_fn(&HarensRE::ChunkingResultProc), this);
	tChunkHashing = thread(std::mem_fn(&HarensRE::ChunkHashing), this);
	for (int i = 0; i < reducerNum; ++i)
	{
		tChunkMatch[i] = thread(std::mem_fn(&HarensRE::ChunkMatch), this);
	}
}

void HarensRE::Stop()
{
	// send termination signal
	IO::Print("seding terminaltion signial to kernel...\n");
	unique_lock<mutex> terminateSigLock(terminateSigMutex);
	terminateSig = true;
	terminateSigLock.unlock();

	// wait for the kernel to terminate
	IO::Print("redundancy elimination module kernel is going to terminate...\n");
	tReadData.join();
	tChunkingKernel.join();
	tChunkingResultProc.join();
	tChunkHashing.join();
	for (int i = 0; i < reducerNum; ++i)
		tChunkMatch[i].join();

	// make conclusion
	IO::Print("Chunking kernel time: %f ms\n", timeChunkingKernel);
	IO::Print("Chunking processing time: %f ms\n", timeChunkPartitioning);
	IO::Print("Map (Chunk hashing) time: %f ms\n", timeChunkHashing);
	IO::Print("Reduce (Chunk matching) time %f ms\n", timeChunkMatching);
	for (int i = 0; i < reducerNum; ++i)
		totalDuplicationSize += duplicationSize[i];
	IO::Print("Found %s of redundency, "
		, IO::InterpretSize(totalDuplicationSize));
	IO::Print("which is %f %% of file\n"
		, (float)totalDuplicationSize / totalFileLen * 100);
}

//TODO: start here
//Use iterator to walk through requestQueue
//remember to notifyone() when this request is done
//figure out how to know a request is done, especially when input file is too large for one pagable memory
void 
HarensRE::ReadData() 
{
	// variables available in the whole scope
	int count = 0;
	FixedSizedCharArray charArrayBuffer(MAX_BUFFER_LEN);
	int pagableBufferIdx = 0;

	// this function would only end when it received termination signals
	while (true)
	{
		// if request queue is empty, check for termination signal
		// terminate reading data process if termination signal received
		// otherwise, wait for new request
		unique_lock<mutex> requestQueueLock(requestQueueMutex);
		while (requestQueue.empty())
		{
			unique_lock<mutex> terminateSigLock(terminateSigMutex);
			if (terminateSig)
			{
				IO::Print("Total file size: %s\n", IO::InterpretSize(totalFileLen));
				IO::Print("Need %d pagable buffers\n", count);
				return;
			}
			terminateSigLock.unlock();
			newRequestCond.wait(requestQueueLock);
		}

		// get the request that came first
		auto& reqResCond = requestQueue.front();
		requestQueue.pop();
		string& request = get<0>(reqResCond);
		auto& result = get<1>(reqResCond);
		int& resultLenInUint8 = get<2>(reqResCond);
		mutex& resultMutex = get<3>(reqResCond);
		condition_variable& resultCond = get<4>(reqResCond);

		// set a vector containing all the indices of pagable buffers used to store this data
		vector<int> pagableBuffersUsed;

		// set up the overlap buffer
		char overlap[WINDOW_SIZE - 1];

		// read the first part
		unique_lock<mutex> readFileInitLock(pagableBufferMutex[pagableBufferIdx]);
		while (pagableBufferObsolete[pagableBufferIdx] == false) 
		{
			pagableBufferCond[pagableBufferIdx].wait(readFileInitLock);
		}
		startReading = clock();
		IO::fileReader->SetupReader(strdup(request.c_str()));	
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);

		++count;
		memcpy(pagableBuffer[pagableBufferIdx], 
			   charArrayBuffer.GetArr(), 
			   pagableBufferLen[pagableBufferIdx]);
		pagableBufferLen[pagableBufferIdx] = charArrayBuffer.GetLen();
		totalFileLen += pagableBufferLen[pagableBufferIdx];
		pagableBufferObsolete[pagableBufferIdx] = false;

		// copy the last window into overlap
		memcpy(overlap, 
			   &pagableBuffer[pagableBufferIdx]
			   				  [pagableBufferLen[pagableBufferIdx] - WINDOW_SIZE + 1], 
			   WINDOW_SIZE - 1);
		readFileInitLock.unlock();
		pagableBuffersUsed.push_back(pagableBufferIdx);
		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		endReading = clock();
		timeReading += (endReading - startReading) * 1000 / CLOCKS_PER_SEC;
		
		// read the rest, this block only ends when reading to the end of file
		while (true) 
		{
			unique_lock<mutex> readFileIterLock(pagableBufferMutex[pagableBufferIdx]);
			while (pagableBufferObsolete[pagableBufferIdx] == false) 
			{
				pagableBufferCond[pagableBufferIdx].wait(readFileIterLock);
			}
			startReading = clock();

			IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);

			// read the end of file
			if (charArrayBuffer.GetLen() == 0) 
			{
				readFileIterLock.unlock();
				break;	
			}
			++count;
			// copy the overlap into current part
			memcpy(pagableBuffer[pagableBufferIdx], 
				   overlap, 
				   WINDOW_SIZE - 1);		
			memcpy(&pagableBuffer[pagableBufferIdx][WINDOW_SIZE - 1], 
				   charArrayBuffer.GetArr(), 
				   charArrayBuffer.GetLen());
			pagableBufferLen[pagableBufferIdx] = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
			totalFileLen += charArrayBuffer.GetLen();
			pagableBufferObsolete[pagableBufferIdx] = false;

			// copy the last window into overlap
			memcpy(overlap, 
				   &pagableBuffer[pagableBufferIdx][charArrayBuffer.GetLen()], 
				   WINDOW_SIZE - 1);	
			readFileIterLock.unlock();
			pagableBuffersUsed.push_back(pagableBufferIdx);
			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			endReading = clock();
			timeReading += (endReading - startReading) * 1000 / CLOCKS_PER_SEC;
		}

		// done reading data for this request
		unique_lock<mutex> packageQueueLock(packageQueueMutex);
		// only pass result and resultCond by reference, because pagableBuffersUsed \
		// would be destroyed by the end of processing this request
		packageQueue.push(make_tuple(pagableBuffersUsed,
									 ref(result),
									 ref(resultLenInUint8),
								 	 ref(resultMutex),
									 ref(resultCond)));
		packageQueueLock.unlock();
		newPackageCond.notify_all();
	}
}

void 
HarensRE::ChunkingKernel() 
{
	// variables available in the whole scope
	int fixedBufferIdx = 0;
	int streamIdx = 0;

	// this function would only end when it received termination signals
	while (true) 
	{
		// if package queue is empty, check for termination signal
		// terminate chunking kernel process if termination signal is received 
		// otherwise, wait for new package
		unique_lock<mutex> packageQueueLock(packageQueueMutex);
		while (packageQueue.empty())
		{
			unique_lock<mutex> terminateSigLock(terminateSigMutex);
			if (terminateSig)
			{
				return;
			}
			terminateSigLock.unlock();
			newPackageCond.wait(packageQueueLock);
		}

		// get the package that came first
		auto& pkgResCond = packageQueue.front();
		packageQueue.pop();
		vector<int> pagableBuffersUsed = get<0>(pkgResCond);
		auto& result = get<1>(pkgResCond);
		int& resultLenInUint8 = get<2>(pkgResCond);
		mutex& resultMutex = get<3>(pkgResCond);
		condition_variable& resultCond = get<4>(pkgResCond);

		// set a vector containing all the indices of result-host used to store the results
		vector<int> resultHostUsed;

		for(vector<int>::iterator pagableBufferIdx = pagableBuffersUsed.begin();
			pagableBufferIdx != pagableBuffersUsed.end();
			++pagableBufferIdx)
		{
			// pagable buffer is ready since this thread has received the package

			// get result host ready
			unique_lock<mutex> resultHostLock(hostResultMutex[fixedBufferIdx]);
			while (hostResultExecuting[fixedBufferIdx] == true) 
			{
				hostResultCond[fixedBufferIdx].wait(resultHostLock);
			}

			startChunkingKernel = clock();
			fixedBufferLen[fixedBufferIdx] = pagableBufferLen[*pagableBufferIdx];
			memcpy(fixedBuffer[fixedBufferIdx], 
				   pagableBuffer[*pagableBufferIdx], 
				   fixedBufferLen[fixedBufferIdx]);
			//pagable buffer is still not obsolete here!

			re.RabinHashAsync(kernelInputBuffer[fixedBufferIdx], 
							  fixedBuffer[fixedBufferIdx], 
							  fixedBufferLen[fixedBufferIdx],
							  kernelResultBuffer[fixedBufferIdx], 
							  hostResultBuffer[fixedBufferIdx],
							  stream[streamIdx]);

			hostResultLen[fixedBufferIdx] = fixedBufferLen[fixedBufferIdx] - WINDOW_SIZE + 1;
			hostResultExecuting[fixedBufferIdx] = true;
			resultHostLock.unlock();
			resultHostUsed.push_back(fixedBufferIdx);
			fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
			streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
			endChunkingKernel = clock();
			timeChunkingKernel += (endChunkingKernel - startChunkingKernel) * 1000 / CLOCKS_PER_SEC;
		}

		// done (Rabin) hashing data for this package
		unique_lock<mutex> rabinQueueLock(rabinQueueMutex);
		// only pass result and resultCond by reference, because resultHostUsed \
		// would be destroyed by the end of processing this package
		rabinQueue.push(make_tuple(resultHostUsed,
								   ref(result),
								   ref(resultLenInUint8),
								   ref(resultMutex),
								   ref(resultCond)));
		rabinQueueLock.unlock();
		newRabinCond.notify_all();
	}
}

void 
HarensRE::ChunkingResultProc() 
{
	// variables available in the whole scope
	int streamIdx = 0;
	
	// this function would only end when it received termination signals 
	while (true) 
	{
		// if rabin hash result queue is empty, check for termination signal
		// terminate chunking result processing if termination signal is received
		// otherwise, wait for new package
		unique_lock<mutex> rabinQueueLock(rabinQueueMutex);
		while (rabinQueue.empty())
		{
			unique_lock<mutex> terminateSigLock(terminateSigMutex);
			if (terminateSig)
			{
				return;
			}
			terminateSigLock.unlock();
			newRabinCond.wait(rabinQueueLock);
		}

		// get the rabin hash result that came first
		auto& rabinResCond = rabinQueue.front();
		rabinQueue.pop();
		vector<int> resultHostUsed = get<0>(rabinResCond);
		auto& result = get<1>(rabinResCond);
		int& resultLenInUint8 = get<2>(rabinResCond);
		mutex& resultMutex = get<3>(rabinResCond);
		condition_variable& resultCond = get<4>(rabinResCond);

		// set a vector containing all the indices of chunking result buffers used
		vector<int> chunkingResultBufferUsed;

		for(vector<int>::iterator resultHostIdx = resultHostUsed.begin();
			resultHostIdx != resultHostUsed.end();
			++resultHostIdx)
		{
			// result host is ready since this thread has received the rabin hash result
			
			// wait until the last stream with the same stream index is finished
			cudaStreamSynchronize(stream[streamIdx]);
			// get the chunking result ready
			unique_lock<mutex> chunkingResultLock(chunkingResultMutex[streamIdx]);
			while (chunkingResultObsolete[streamIdx] == false) 
			{
				chunkingResultCond[streamIdx].wait(chunkingResultLock);
			}
				
			startChunkPartitioning = clock();
			
			int chunkingResultIdx = 0;
			unsigned int resultHostLen = hostResultLen[*resultHostIdx];
			for (unsigned int j = 0; j < resultHostLen; ++j) 
			{
				if (hostResultBuffer[*resultHostIdx][j] == 0) 
				{
					chunkingResultBuffer[streamIdx][chunkingResultIdx++] = j;
				}
			}

			chunkingResultLen[streamIdx] = chunkingResultIdx;

			unique_lock<mutex> resultHostLock(hostResultMutex[*resultHostIdx]);
			hostResultExecuting[*resultHostIdx] = false;
			resultHostLock.unlock();
			hostResultCond[*resultHostIdx].notify_one();
			chunkingResultObsolete[streamIdx] = false;
			chunkingResultLock.unlock();

			chunkingResultBufferUsed.push_back(streamIdx);
			streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
			endChunkPartitioning = clock();
			timeChunkPartitioning += (endChunkPartitioning - startChunkPartitioning) * 1000 / CLOCKS_PER_SEC;
		}
		
		// done chunking the package
		unique_lock<mutex> chunkQueueLock(chunkQueueMutex);
		// only pass result and resultCond by reference, because chunkingResultBufferUsed \
		// would be destroyed by the end of processing this package chunking
		chunkQueue.push(make_tuple(chunkingResultBufferUsed,
								   ref(result),
								   ref(resultLenInUint8),
								   ref(resultMutex),
								   ref(resultCond)));
		chunkQueueLock.unlock();
		newChunksCond.notify_all();
	}
}

void 
HarensRE::ChunkHashing() 
{
	// variables available in the whole scope
	int pagableBufferIdx = 0;

	// this function would only end when it received termination signals 
	while (true) 
	{
		// if chunking result queue is empty, check for termination signal
		// terminate chunk hashing if termination signal is received
		// otherwise, wait for new package
		unique_lock<mutex> chunkQueueLock(chunkQueueMutex);
		while (chunkQueue.empty())
		{
			unique_lock<mutex> terminateSigLock(terminateSigMutex);
			if (terminateSig)
			{
				return;
			}
			terminateSigLock.unlock();
			newChunksCond.wait(chunkQueueLock);
		}

		// get the chunking result that came first
		auto& chunksResCond = chunkQueue.front();
		chunkQueue.pop();
		vector<int> chunkingResultBufferUsed = get<0>(chunksResCond);
		auto& result = get<1>(chunksResCond);
		int& resultLenInUint8 = get<2>(chunksResCond);
		mutex& resultMutex = get<3>(chunksResCond);
		condition_variable& resultCond = get<4>(chunksResCond);

		for(vector<int>::iterator chunkingResultIdx = chunkingResultBufferUsed.begin();
			chunkingResultIdx != chunkingResultBufferUsed.end();
			++chunkingResultIdx)
		{
			// chunking result is ready since this thread has received the result

			// pagable buffer is ready because it's never released

			startChunkHashing = clock();
			for (int i = 0; i < mapperNum; ++i) 
			{
				segmentThreads[i] = thread(std::mem_fn(&HarensRE::ChunkSegmentHashing), 
											this, 
											pagableBufferIdx, 
											*chunkingResultIdx, 
											i, 
											result, 
											ref(resultLenInUint8), 
											ref(resultMutex));
			}

			for (int i = 0; i < mapperNum; ++i) 
			{
				segmentThreads[i].join();
			}

			unique_lock<mutex> pagableLock(pagableBufferMutex[pagableBufferIdx]);
			pagableBufferObsolete[pagableBufferIdx] = true;
			pagableLock.unlock();
			pagableBufferCond[pagableBufferIdx].notify_one();
			unique_lock<mutex> chunkingResultLock(chunkingResultMutex[*chunkingResultIdx]);
			chunkingResultObsolete[*chunkingResultIdx] = true;
			chunkingResultLock.unlock();
			chunkingResultCond[*chunkingResultIdx].notify_one();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			endChunkHashing = clock();
			timeChunkHashing += (endChunkHashing - startChunkHashing) * 1000 / CLOCKS_PER_SEC;
		}
		
		// done computing hash values for the chunks
		unique_lock<mutex> hashQueueLock(hashQueueMutex);
		hashQueue.push(make_tuple(ref(result),
								  ref(resultLenInUint8),
								  ref(resultCond)));
		hashQueueLock.unlock();
		newHashCond.notify_all();
	}
}

void 
HarensRE::ChunkSegmentHashing(int pagableBufferIdx, 
							  int chunkingResultIdx, 
							  int segmentNum,
							  vector< tuple<int, unsigned char*, int, char*> >* result,
							  int& resultLenInUint8,
							  mutex& resultMutex) 
{
	int listSize = chunkingResultLen[chunkingResultIdx];
	unsigned int* chunkingResultSeg = &chunkingResultBuffer[chunkingResultIdx]
													  [segmentNum * listSize / mapperNum];
	int segLen = listSize / mapperNum;
	if ((segmentNum + 1) * listSize / mapperNum > listSize)
	{
		segLen = listSize - segmentNum * listSize / mapperNum;
	}
	re.ChunkHashingAsync(chunkingResultSeg, 
						 segLen, 
						 pagableBuffer[pagableBufferIdx],
						 result,
						 resultLenInUint8,
						 resultMutex);
}

void 
HarensRE::ChunkMatch() 
{
	// variables available in the whole scope
	unsigned char* toBeDel = nullptr;

	// this function would only end when it received termination signals 
	while (true) 
	{
		// if hash queue is empty, check for termination signal
		// terminate chunk match processing if termination signal is received
		// otherwise, wait for new hash coming
		unique_lock<mutex> hashQueueLock(hashQueueMutex);
		while (hashQueue.empty())
		{
			unique_lock<mutex> terminateSigLock(terminateSigMutex);
			if (terminateSig)
			{
				return;
			}
			terminateSigLock.unlock();
			newHashCond.wait(hashQueueLock);
		}

		// get the (SHA1) hash values of chunks that came first
		auto& resCond = hashQueue.front();
		hashQueue.pop();
		auto& result = get<0>(resCond);
		int& resultLenInUint8 = get<1>(resCond);
		condition_variable& resultCond = get<2>(resCond);
			
		startChunkMatching = clock();

		for(auto resultIter = result->begin();
			resultIter != result->end();
			++resultIter)
		{
			// get the values 
			int hashLen = get<0>(*resultIter); 
			unsigned char* hashVal = get<1>(*resultIter);
			int chunkLen = get<2>(*resultIter);
			char* chunkVal = get<3>(*resultIter);

			// find out the index of circHashPool to handle this hash value
			unsigned int hashPoolIdx;
			memcpy(&hashPoolIdx, hashVal, sizeof(unsigned int));
			hashPoolIdx %= reducerNum;

			// find out duplications
			bool found;
			circHashPoolMutex[hashPoolIdx].lock();
			found = circHashPool[hashPoolIdx].FindAndAdd(hashVal, toBeDel);
			circHashPoolMutex[hashPoolIdx].unlock();
			if (found)
			{
				duplicationSize[hashPoolIdx] += chunkLen;
				resultLenInUint8 -= chunkLen;
				get<2>(*resultIter) = 0;
				delete[] chunkVal;
				get<3>(*resultIter) = NULL;
				if (toBeDel != nullptr) 
				{	
					// remove chunk corresponding to toBeDel from storage
				}
			}
		}

		endChunkMatching = clock();
		timeChunkMatching += (endChunkMatching - startChunkMatching) * 1000 / CLOCKS_PER_SEC;
		hashQueueLock.unlock();
		resultCond.notify_one();
	}
}