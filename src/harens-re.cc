#include "harens-re.h"
using namespace std;

HarensRE::HarensRE(int mapperNum, int reducerNum) 
	: charArrayBuffer(MAX_BUFFER_LEN), chunk_hash_queue_pool(reducerNum) 
{
	this->mapperNum = mapperNum;
	this->reducerNum = reducerNum;
	segment_threads = new thread[mapperNum];
	chunk_match_threads = new thread[reducerNum];
	circ_hash_pool = new LRUStrHash<SHA_DIGEST_LENGTH>[reducerNum];
	duplication_size = new unsigned long long[reducerNum];
	for (int i = 0; i < reducerNum; ++i) 
	{
		circ_hash_pool[i] = LRUStrHash<SHA_DIGEST_LENGTH>(MAX_CHUNK_NUM / reducerNum);
		duplication_size[i] = 0;
	}

	re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::NonMultifingerprint);
	// initialize pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) 
	{
		pagable_buffer[i] = new char[MAX_BUFFER_LEN];
		pagable_buffer_obsolete[i] = true;
	}
	// initialize fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaMallocHost((void**)&fixed_buffer[i], MAX_BUFFER_LEN);
		// fixed_buffer_obsolete[i] = true;
	}
	// initialize chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaMalloc((void**)&input_kernel[i], MAX_BUFFER_LEN);
		cudaMalloc((void**)&result_kernel[i], MAX_WINDOW_NUM * BYTES_IN_UINT);
		cudaMallocHost((void**)&result_host[i], MAX_WINDOW_NUM * BYTES_IN_UINT);
		result_host_obsolete[i] = true;
		result_host_executing[i] = false;
	}
	// initialize chunking result processing
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) 
	{
		cudaStreamCreate(&stream[i]);
		chunking_result[i] = new unsigned int[MAX_WINDOW_NUM];
		chunking_result_obsolete[i] = true;
	}
}

HarensRE::~HarensRE() 
{
	delete[] segment_threads;
	delete[] chunk_match_threads;
	delete[] circ_hash_pool;
	delete[] duplication_size;

	// destruct chunking result proc
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) 
	{
		cudaStreamDestroy(stream[i]);
		delete[] chunking_result[i];
	}
	// destruct chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaFree(input_kernel[i]);
		cudaFree(result_kernel[i]);
		cudaFreeHost(result_host[i]);
	}
	// destruct fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) 
	{
		cudaFreeHost(fixed_buffer[i]);
	}
	// destruct pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) 
	{
		delete[] pagable_buffer[i];
	}
}

vector< tuple<int, unsigned char*, int, char*> >* 
HarensRE::HandleGetRequest(string request)
{
	// put values into request list as reference
	mutex resultMutex;
	condition_variable resultCond;
	vector< tuple<int, unsigned char*, int, char*> >*  result
		= new vector< tuple<int, unsigned char*, int, char*> >();
	unique_lock<mutex> requestQueueLock(requestQueueMutex);
	requestQueue.push(make_tuple(ref(request), 
								 ref(result), 
								 ref(resultCond)));
	requestQueueLock.unlock();
	newRequestCond.notify_all();
	// wait for result notification
	unique_lock<mutex> resultLock(resultMutex);
	resultCond.wait(resultLock, []{return result->size() > 0;});

	return result;
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
		chunk_match_threads[i] = thread(std::mem_fn(&HarensRE::ChunkMatch), this, i);
	}
}

void HarensRE::End()
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
		chunk_match_threads[i].join();

	// make conclusion
	IO::Print("Chunking kernel time: %f ms\n", time_ck);
	IO::Print("Chunking processing time: %f ms\n", time_cp);
	IO::Print("Map (Chunk hashing) time: %f ms\n", time_ch);
	IO::Print("Reduce (Chunk matching) time %f ms\n", time_cm);
	IO::Print("Total time: %f ms\n", time_tot);
	for (int i = 0; i < reducerNum; ++i)
		total_duplication_size += duplication_size[i];
	IO::Print("Found %s of redundency, "
		, IO::InterpretSize(total_duplication_size));
	IO::Print("which is %f %% of file\n"
		, (float)total_duplication_size / file_length * 100);
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
	unsigned long long fileLength = 0;
	FixedSizeCharArray charArrayBuffer;
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
				IO::Print("Total file size: %s\n", IO::InterpretSize(fileLength));
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
		condition_variable& resultCond = get<2>(reqResCond);

		// set a vector containing all the indices of pagable buffers used to store this data
		vector<int> pagableBuffersUsed;

		// set up the overlap buffer
		char overlap[WINDOW_SIZE - 1];

		// read the first part
		unique_lock<mutex> readFileInitLock(pagable_buffer_mutex[pagableBufferIdx]);
		start_r = clock();
		IO::fileReader->SetupFile(request.c_str());	
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);


		++count;
		memcpy(pagable_buffer[pagableBufferIdx], 
			   charArrayBuffer.GetArr(), 
			   pagable_buffer_len[pagableBufferIdx]);
		pagable_buffer_len[pagableBufferIdx] = charArrayBuffer.GetLen();
		file_length += pagable_buffer_len[pagableBufferIdx];
		pagable_buffer_obsolete[pagableBufferIdx] = false;

		// copy the last window into overlap
		memcpy(overlap, 
			   &pagable_buffer[pagableBufferIdx]
			   				  [pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1], 
			   WINDOW_SIZE - 1);
		readFileInitLock.unlock();
		pagableBuffersUsed.push_back(pagableBufferIdx);
		pagable_buffer_cond[pagableBufferIdx].notify_one();
		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		end_r = clock();
		time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		
		// read the rest, this block only ends when reading to the end of file
		while (true) 
		{
			unique_lock<mutex> readFileIterLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == false) 
			{
				pagable_buffer_cond[pagableBufferIdx].wait(readFileIterLock);
			}
			start_r = clock();

			IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);

			// read nothing
			if (charArrayBuffer.GetLen() == 0) 
			{
				readFileIterLock.unlock();
				pagable_buffer_cond[pagableBufferIdx].notify_all();
				break;	
			}
			++count;
			// copy the overlap into current part
			memcpy(pagable_buffer[pagableBufferIdx], 
				   overlap, 
				   WINDOW_SIZE - 1);		
			memcpy(&pagable_buffer[pagableBufferIdx][WINDOW_SIZE - 1], 
				   charArrayBuffer.GetArr(), 
				   charArrayBuffer.GetLen());
			pagable_buffer_len[pagableBufferIdx] = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
			file_length += charArrayBuffer.GetLen();
			pagable_buffer_obsolete[pagableBufferIdx] = false;

			// copy the last window into overlap
			memcpy(overlap, 
				   &pagable_buffer[pagableBufferIdx][charArrayBuffer.GetLen()], 
				   WINDOW_SIZE - 1);	
			readFileIterLock.unlock();
			pagableBuffersUsed.push_back(pagableBufferIdx);
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			end_r = clock();
			time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		}

		// done reading data for this request
		unique_lock<mutex> packageQueueLock(packageQueueMutex);
		// only pass result and resultCond by reference, because pagableBuffersUsed \
		// would be destroyed by the end of processing this request
		packageQueue.push(make_tuple(pagableBuffersUsed,
									 ref(result),
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
		condition_variable& resultCond = get<2>(pkgResCond);

		// set a vector containing all the indices of result-host used to store the results
		vector<int> resultHostUsed;

		for(vector<int>::iterator pagableBufferIdx = pagableBuffersUsed.begin();
			pagableBufferIdx != pagableBuffersUsed.end();
			++pagableBufferIdx)
		{
			// pagable buffer is ready since this thread has received the package

			// get result host ready
			unique_lock<mutex> resultHostLock(result_host_mutex[fixedBufferIdx]);
			while (result_host_executing[fixedBufferIdx] == true) 
			{
				result_host_cond[fixedBufferIdx].wait(resultHostLock);
			}

			start_ck = clock();
			fixed_buffer_len[fixedBufferIdx] = pagable_buffer_len[*pagableBufferIdx];
			memcpy(fixed_buffer[fixedBufferIdx], 
				   pagable_buffer[*pagableBufferIdx], 
				   fixed_buffer_len[fixedBufferIdx]);
			//pagable buffer is still not obsolete here!

			re.RabinHashAsync(input_kernel[fixedBufferIdx], 
							  fixed_buffer[fixedBufferIdx], 
							  fixed_buffer_len[fixedBufferIdx],
							  result_kernel[fixedBufferIdx], 
							  result_host[fixedBufferIdx],
							  stream[streamIdx]);

			result_host_len[fixedBufferIdx] = fixed_buffer_len[fixedBufferIdx] - WINDOW_SIZE + 1;
			result_host_executing[fixedBufferIdx] = true;
			resultHostLock.unlock();
			resultHostUsed.push_back(fixedBufferIdx);
			result_host_cond[fixedBufferIdx].notify_one();
			fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
			streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
			end_ck = clock();
			time_ck += (end_ck - start_ck) * 1000 / CLOCKS_PER_SEC;
		}

		// done (Rabin) hashing data for this package
		unique_lock<mutex> rabinQueueLock(rabinQueueMutex);
		// only pass result and resultCond by reference, because resultHostUsed \
		// would be destroyed by the end of processing this package
		rabinQueue.push(make_tuple(resultHostUsed,
								   ref(result),
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
		condition_variable& resultCond = get<2>(rabinResCond);

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
			unique_lock<mutex> chunkingResultLock(chunking_result_mutex[streamIdx]);
			while (chunking_result_obsolete[streamIdx] == false) 
			{
				chunking_result_cond[streamIdx].wait(chunkingResultLock);
			}
				
			start_cp = clock();
			
			int chunkingResultIdx = 0;
			unsigned int resultHostLen = result_host_len[*resultHostIdx];
			for (unsigned int j = 0; j < resultHostLen; ++j) 
			{
				if (result_host[*resultHostIdx][j] == 0) 
				{
					chunking_result[streamIdx][chunkingResultIdx++] = j;
				}
			}

			chunking_result_len[streamIdx] = chunkingResultIdx;

			result_host_executing[*resultHostIdx] = false;
			chunking_result_obsolete[streamIdx] = false;
			resultHostLock.unlock();
			result_host_cond[*resultHostIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[streamIdx].notify_one();

			chunkingResultBufferUsed.push_back(streamIdx);
			streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
			end_cp = clock();
			time_cp += (end_cp - start_cp) * 1000 / CLOCKS_PER_SEC;
		}
		
		// done chunking the package
		unique_lock<mutex> chunkQueueLock(chunkQueueMutex);
		// only pass result and resultCond by reference, because chunkingResultBufferUsed \
		// would be destroyed by the end of processing this package chunking
		chunkQueue.push(make_tuple(chunkingResultBufferUsed,
								   ref(result),
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
		condition_variable& resultCond = get<2>(chunksResCond);

		for(vector<int>::iterator chunkingResultIdx = chunkingResultBufferUsed.begin();
			chunkingResultIdx != chunkingResultBufferUsed.end();
			++chunkingResultIdx)
		{
			//Get pagable buffer ready
			unique_lock<mutex> pagableLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == true) 
			{
				unique_lock<mutex> chukingProcEndLock(chunking_proc_end_mutex);
				if (chunking_proc_end) 
				{
					unique_lock<mutex> chunkHashingEndLock(chunk_hashing_end_mutex);
					chunk_hashing_end = true;
					return;
				}
				chukingProcEndLock.unlock();
				pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
			}
			//Get the chunking result ready
			unique_lock<mutex> chunkingResultLock(chunking_result_mutex[*chunkingResultIdx]);
			while (chunking_result_obsolete[*chunkingResultIdx] == true) 
			{
				unique_lock<mutex> chukingProcEndLock(chunking_proc_end_mutex);
				if (chunking_proc_end) 
				{
					unique_lock<mutex> chunkHashingEndLock(chunk_hashing_end_mutex);
					chunk_hashing_end = true;
					return;
				}
				chukingProcEndLock.unlock();
				chunking_result_cond[*chunkingResultIdx].wait(chunkingResultLock);
			}

			start_ch = clock();
			for (int i = 0; i < mapperNum; ++i) 
			{
				segment_threads[i] = thread(std::mem_fn(&Harens::ChunkSegmentHashing)
					, this, pagableBufferIdx, *chunkingResultIdx, i);
			}

			for (int i = 0; i < mapperNum; ++i) 
			{
				segment_threads[i].join();
			}

			pagable_buffer_obsolete[pagableBufferIdx] = true;
			chunking_result_obsolete[*chunkingResultIdx] = true;
			pagableLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[*chunkingResultIdx].notify_one();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			end_ch = clock();
			time_ch += (end_ch - start_ch) * 1000 / CLOCKS_PER_SEC;
		}

		
	}
}

void 
HarensRE::ChunkSegmentHashing(int pagableBufferIdx, int chunkingResultIdx, int segmentNum) 
{
	int listSize = chunking_result_len[chunkingResultIdx];
	unsigned int* chunkingResultSeg = &chunking_result[chunkingResultIdx]
													  [segmentNum * listSize / mapperNum];
	int segLen = listSize / mapperNum;
	if ((segmentNum + 1) * listSize / mapperNum > listSize)
	{
		segLen = listSize - segmentNum * listSize / mapperNum;
	}
	re.ChunkHashingAsync(chunkingResultSeg, 
						 segLen, 
						 pagable_buffer[pagableBufferIdx],
						 chunk_hash_queue_pool);
}

void 
HarensRE::ChunkMatch(int hashPoolIdx) 
{
	unsigned char* toBeDel = nullptr;
	while (true) 
	{
		if (chunk_hash_queue_pool.IsEmpty(hashPoolIdx)) 
		{
			unique_lock<mutex> chunkHashingEndLock(chunk_hashing_end_mutex);
			if (chunk_hashing_end) 
			{
				return;
			}
			else 
			{
				chunkHashingEndLock.unlock();
				this_thread::sleep_for(std::chrono::milliseconds(500));
				continue;
			}
		}
			
		start_cm = clock();

		tuple<unsigned char*, unsigned int> valLenPair = chunk_hash_queue_pool.Pop(hashPoolIdx);
		if (circ_hash_pool[hashPoolIdx].FindAndAdd(get<0>(valLenPair), toBeDel))
			duplication_size[hashPoolIdx] += get<1>(valLenPair);
		if (toBeDel != nullptr) 
		{
			//Remove chunk corresponding to toBeDel from storage
		}

		end_cm = clock();
		time_cm += (end_cm - start_cm) * 1000 / CLOCKS_PER_SEC;
	}
}