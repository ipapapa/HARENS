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
						   int* resultLenInUint8)
{
	// put values into request list as reference
	mutex* resultMutex = new mutex();
	condition_variable* resultCond = new condition_variable();
	*resultLenInUint8 = 0;
	unique_lock<mutex> requestQueueLock(requestQueueMutex);
	unique_lock<mutex> resultQueueLock(resultQueueMutex);
	requestQueue.push(request); 
	resultQueue.push(make_tuple(result, 
								resultLenInUint8,
								resultMutex,
								resultCond));
	resultQueueLock.unlock();
	requestQueueLock.unlock();
	newRequestCond.notify_all();
	// wait for result notification
	unique_lock<mutex> resultLock(*resultMutex);
	resultCond->wait(resultLock, [result]{return result->size() > 0;});
	// clean up
	delete resultMutex;
	delete resultCond;
}

void
HarensRE::Start()
{
	IO::Print("Redundancy elimination module kernel started...\n");
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
	IO::Print("Sending terminaltion signial to kernel...\n");
	unique_lock<mutex> terminateSigLock(terminateSigMutex);
	terminateSig = true;
	terminateSigLock.unlock();

	// wait for the kernel to terminate
	IO::Print("Redundancy elimination module kernel is going to terminate...\n");
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
		string request = requestQueue.front();
		requestQueue.pop();
		requestQueueLock.unlock();

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

		// tell next worker about new pakage coming
		unique_lock<mutex> packageQueueLock(packageQueueMutex);
		packageQueue.push(pagableBufferIdx);
		packageQueueLock.unlock();
		newPackageCond.notify_all();

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

			// tell next worker about new pakage coming
			packageQueueLock.lock();
			packageQueue.push(pagableBufferIdx);
			packageQueueLock.unlock();
			newPackageCond.notify_all();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			endReading = clock();
			timeReading += (endReading - startReading) * 1000 / CLOCKS_PER_SEC;
		}

		// done reading data for this request
		packageQueueLock.lock();
		packageQueue.push(-1);	// -1 is the deliminator between packages
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
		// pagable buffer is ready since this thread has received the package
		int pagableBufferIdx = packageQueue.front();
		packageQueue.pop();
		packageQueueLock.unlock();

		// if found a deliminator
		if (pagableBufferIdx == -1)
		{
			// done (Rabin) hashing data for current package
			unique_lock<mutex> rabinQueueLock(rabinQueueMutex);
			rabinQueue.push(-1);	// -1 is the deliminator between packages
			rabinQueueLock.unlock();
			newRabinCond.notify_all();
			continue;
		}

		// get result host ready
		unique_lock<mutex> resultHostLock(hostResultMutex[fixedBufferIdx]);
		while (hostResultExecuting[fixedBufferIdx] == true) 
		{
			hostResultCond[fixedBufferIdx].wait(resultHostLock);
		}

		startChunkingKernel = clock();
		fixedBufferLen[fixedBufferIdx] = pagableBufferLen[pagableBufferIdx];
		memcpy(fixedBuffer[fixedBufferIdx], 
			   pagableBuffer[pagableBufferIdx], 
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
		
		// done (Rabin) hashing data for this buffer page
		unique_lock<mutex> rabinQueueLock(rabinQueueMutex);
		rabinQueue.push(fixedBufferIdx);
		rabinQueueLock.unlock();
		newRabinCond.notify_all();

		fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
		streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
		endChunkingKernel = clock();
		timeChunkingKernel += (endChunkingKernel - startChunkingKernel) * 1000 / CLOCKS_PER_SEC;
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
		// result host is ready since this thread has received the rabin hash result
		int resultHostIdx = rabinQueue.front();
		rabinQueue.pop();
		rabinQueueLock.unlock();

		// if found a deliminator
		if (resultHostIdx == -1)
		{
			// done chunking the package
			unique_lock<mutex> chunkQueueLock(chunkQueueMutex);
			chunkQueue.push(-1);
			chunkQueueLock.unlock();
			newChunksCond.notify_all();
			continue;
		}
		
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
		unsigned int resultHostLen = hostResultLen[resultHostIdx];
		for (unsigned int j = 0; j < resultHostLen; ++j) 
		{
			if (hostResultBuffer[resultHostIdx][j] == 0) 
			{
				chunkingResultBuffer[streamIdx][chunkingResultIdx++] = j;
			}
		}

		chunkingResultLen[streamIdx] = chunkingResultIdx;

		unique_lock<mutex> resultHostLock(hostResultMutex[resultHostIdx]);
		hostResultExecuting[resultHostIdx] = false;
		resultHostLock.unlock();
		hostResultCond[resultHostIdx].notify_one();
		chunkingResultObsolete[streamIdx] = false;
		chunkingResultLock.unlock();

		// done chunking the buffer page
		unique_lock<mutex> chunkQueueLock(chunkQueueMutex);
		chunkQueue.push(streamIdx);
		chunkQueueLock.unlock();
		newChunksCond.notify_all();

		streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
		endChunkPartitioning = clock();
		timeChunkPartitioning += (endChunkPartitioning - startChunkPartitioning) * 1000 / CLOCKS_PER_SEC;
	}
}

void 
HarensRE::ChunkHashing() 
{
	// variables available in the whole scope
	int pagableBufferIdx = 0;
	vector< tuple<int, unsigned char*, int, char*> >* result = nullptr;
	int* resultLenInUint8;
	mutex* resultMutex;
	condition_variable* resultCond;

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
		// chunking result is ready since this thread has received the result
		// pagable buffer is ready because it's never released
		int chunkingResultIdx = chunkQueue.front();
		chunkQueue.pop();
		chunkQueueLock.unlock();

		// if found a deliminator
		if (chunkingResultIdx == -1)
		{
			// done computing hash values for the chunks
			unique_lock<mutex> hashQueueLock(hashQueueMutex);
			hashQueue.push(make_tuple(result,
									  resultLenInUint8,
									  resultMutex,
									  resultCond));
			hashQueueLock.unlock();
			newHashCond.notify_all();
			result = nullptr;
			continue;
		}

		// if dealing with new package
		if (result == nullptr)
		{
			// result queue is not empty because new chunking results coming
			unique_lock<mutex> resultQueueLock(resultQueueMutex);
			auto& curResult = resultQueue.front();
			resultQueue.pop();
			resultQueueLock.unlock();
			result = get<0>(curResult);
			resultLenInUint8 = get<1>(curResult);
			resultMutex = get<2>(curResult);
			resultCond = get<3>(curResult);
		}

		startChunkHashing = clock();
		for (int i = 0; i < mapperNum; ++i) 
		{
			segmentThreads[i] = thread(std::mem_fn(&HarensRE::ChunkSegmentHashing), 
									   this, 
									   pagableBufferIdx, 
									   chunkingResultIdx, 
									   i, 
									   result, 
									   resultLenInUint8, 
									   resultMutex);
		}

		for (int i = 0; i < mapperNum; ++i) 
		{
			segmentThreads[i].join();
		}

		unique_lock<mutex> pagableLock(pagableBufferMutex[pagableBufferIdx]);
		pagableBufferObsolete[pagableBufferIdx] = true;
		pagableLock.unlock();
		pagableBufferCond[pagableBufferIdx].notify_one();
		unique_lock<mutex> chunkingResultLock(chunkingResultMutex[chunkingResultIdx]);
		chunkingResultObsolete[chunkingResultIdx] = true;
		chunkingResultLock.unlock();
		chunkingResultCond[chunkingResultIdx].notify_one();

		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		endChunkHashing = clock();
		timeChunkHashing += (endChunkHashing - startChunkHashing) * 1000 / CLOCKS_PER_SEC;
	}
}

void 
HarensRE::ChunkSegmentHashing(int pagableBufferIdx, 
							  int chunkingResultIdx, 
							  int segmentNum,
							  vector< tuple<int, unsigned char*, int, char*> >* result,
							  int* resultLenInUint8,
							  mutex* resultMutex) 
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
		auto& resLenCond = hashQueue.front();
		hashQueue.pop();
		hashQueueLock.unlock();
		vector< tuple<int, unsigned char*, int, char*> >* result = get<0>(resLenCond);
		int* resultLenInUint8 = get<1>(resLenCond);
		mutex* resultMutex = get<2>(resLenCond);
		condition_variable* resultCond = get<3>(resLenCond);

		bool flag = true;

		resultMutex->lock();
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
			toBeDel = nullptr;
			if (found)
			{
				duplicationSize[hashPoolIdx] += chunkLen;
				*resultLenInUint8 -= chunkLen;
				get<2>(*resultIter) = 0;
				delete[] chunkVal;
				get<3>(*resultIter) = nullptr;
				if (toBeDel != nullptr) 
				{	
					// remove chunk corresponding to toBeDel from storage
				}
			}
		}
		resultMutex->unlock();

		resultCond->notify_one();
	}
}