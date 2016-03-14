#include "Harens.h"
using namespace std;

Harens::Harens(int mapperNum, int reducerNum) 
	: charArrayBuffer(MAX_BUFFER_LEN), chunkHashQueuePool(reducerNum) {
	this->mapperNum = mapperNum;
	this->reducerNum = reducerNum;
	segmentThreads = new thread[mapperNum];
	chunkMatchingThreads = new thread[reducerNum];
	circHashPool = new LRUStrHash<SHA1_HASH_LENGTH>[reducerNum];
	duplicationSize = new unsigned long long[reducerNum];
	for (int i = 0; i < reducerNum; ++i) {
		circHashPool[i] = LRUStrHash<SHA1_HASH_LENGTH>(MAX_CHUNK_NUM / reducerNum);
		duplicationSize[i] = 0;
	}

	re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::NonMultifingerprint);
	//initialize pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) {
		pagableBuffer[i] = new char[MAX_BUFFER_LEN];
		pagableBufferObsolete[i] = true;
	}
	//initialize fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaMallocHost((void**)&fixedBuffer[i], MAX_BUFFER_LEN);
		//fixed_buffer_obsolete[i] = true;
	}
	//initialize chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaMalloc((void**)&kernelInputBuffer[i], MAX_BUFFER_LEN);
		cudaMalloc((void**)&kernelResultBuffer[i], MAX_WINDOW_NUM * BYTES_IN_UINT);
		cudaMallocHost((void**)&hostResultBuffer[i], MAX_WINDOW_NUM * BYTES_IN_UINT);
		hostResultObsolete[i] = true;
		hostResultExecuting[i] = false;
	}
	//initialize chunking result processing
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) {
		cudaStreamCreate(&stream[i]);
		chunkingResultBuffer[i] = new unsigned int[MAX_WINDOW_NUM];
		chunkingResultObsolete[i] = true;
	}
	//initialize chunk hashing
	//for (int i = 0; i < STREAM_NUM; ++i) {
	//	segmentThreads[i] = new thread[FINGERPRINTING_THREAD_NUM];
	//	for (int j = 0; j < FINGERPRINTING_THREAD_NUM; ++j) {
	//		//MAX_WINDOW_NUM / 4 is a guess of the upper bound of the number of chunks
	//		/*chunk_hashing_value_queue[i][j] = LRUUcharArrayQueue(MAX_WINDOW_NUM / 4);
	//		chunk_len_queue[i][j] = LRUUintQueue(MAX_WINDOW_NUM / 4);*/
	//	}
	//}
}

Harens::~Harens() {
	delete[] segmentThreads;
	delete[] chunkMatchingThreads;
	delete[] circHashPool;
	delete[] duplicationSize;

	//destruct chunk hashing & matching
	/*for (int i = 0; i < STREAM_NUM; ++i) {
	delete[] segmentThreads[i];
	}*/
	//destruct chunking result proc
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) {
		cudaStreamDestroy(stream[i]);
		delete[] chunkingResultBuffer[i];
	}
	//destruct chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaFree(kernelInputBuffer[i]);
		cudaFree(kernelResultBuffer[i]);
		cudaFreeHost(hostResultBuffer[i]);
	}
	//destruct fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaFreeHost(fixedBuffer[i]);
	}
	//destruct pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) {
		delete[] pagableBuffer[i];
	}
}
	
int Harens::Execute() {
	IO::Print("\n======= CUDA Implementation With Pipeline and Single Machine MapReduce ======\n");
	
	//Create threads
	thread tReadFile(std::mem_fn(&Harens::ReadFile), this);
	tReadFile.join();
	start = clock();
	//thread tTransfer(Transfer);
	thread tChunkingKernel(std::mem_fn(&Harens::ChunkingKernel), this);
	thread tChunkingResultProc(std::mem_fn(&Harens::ChunkingResultProc), this);
	thread tChunkHashing(std::mem_fn(&Harens::ChunkHashing), this);
	for (int i = 0; i < reducerNum; ++i)
		chunkMatchingThreads[i] = thread(std::mem_fn(&Harens::ChunkMatch), this, i);

	//tTransfer.join();
	tChunkingKernel.join();
	tChunkingResultProc.join();
	tChunkHashing.join();
	for (int i = 0; i < reducerNum; ++i)
		chunkMatchingThreads[i].join();
	//tRoundQuery.join();

	end = clock();
	timeTotal = (end - start) * 1000 / CLOCKS_PER_SEC;
	IO::Print("Read file time: %f ms\n", timeReading);
	//printf("Transfer time: %f ms\n", time_t);
	IO::Print("Chunking kernel time: %f ms\n", timeChunkingKernel);
	IO::Print("Chunking processing time: %f ms\n", timeChunkPartitioning);
	IO::Print("Map (Chunk hashing) time: %f ms\n", timeChunkHashing);
	IO::Print("Reduce (Chunk matching) time %f ms\n", timeChunkMatching);
	IO::Print("Total time: %f ms\n", timeTotal);
	for (int i = 0; i < reducerNum; ++i)
		totalDuplicationSize += duplicationSize[i];
	IO::Print("Found %s of redundency, "
		, IO::InterpretSize(totalDuplicationSize));
	IO::Print("which is %f %% of file\n"
		, (float)totalDuplicationSize / totalFileLen * 100);

	return 0;
}

void Harens::Test(double &rate, double &time) {
	//Create threads
	thread tReadFile(std::mem_fn(&Harens::ReadFile), this);
	tReadFile.join();
	start = clock();
	//thread tTransfer(Transfer);
	thread tChunkingKernel(std::mem_fn(&Harens::ChunkingKernel), this);
	thread tChunkingResultProc(std::mem_fn(&Harens::ChunkingResultProc), this);
	thread tChunkHashing(std::mem_fn(&Harens::ChunkHashing), this);
	for (int i = 0; i < reducerNum; ++i)
		chunkMatchingThreads[i] = thread(std::mem_fn(&Harens::ChunkMatch), this, i);

	//tTransfer.join();
	tChunkingKernel.join();
	tChunkingResultProc.join();
	tChunkHashing.join();
	for (int i = 0; i < reducerNum; ++i)
		chunkMatchingThreads[i].join();
	//tRoundQuery.join();

	end = clock();

	rate = totalDuplicationSize * 100.0 / totalFileLen;
	time = (end - start) * 1000 / CLOCKS_PER_SEC;
}

/*
* Read data from a plain text or pcap file into memory.
* Transfer is done in this step now.
*/
void Harens::ReadFile() {
	int pagableBufferIdx = 0;
	//Read the first part
	unique_lock<mutex> readFileInitLock(pagableBufferMutex[pagableBufferIdx]);
	startReading = clock();
	IO::fileReader->SetupReader(IO::input_file_name);
	IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);
	pagableBufferLen[pagableBufferIdx] = charArrayBuffer.GetLen();
	memcpy(pagableBuffer[pagableBufferIdx], 
		   charArrayBuffer.GetArr(), 
		   pagableBufferLen[pagableBufferIdx]);
	totalFileLen += pagableBufferLen[pagableBufferIdx];
	++count;

	memcpy(overlap, 
		   &pagableBuffer[pagableBufferIdx]
		   				  [pagableBufferLen[pagableBufferIdx] - WINDOW_SIZE + 1], 
		   WINDOW_SIZE - 1);	//copy the last window into overlap
	pagableBufferObsolete[pagableBufferIdx] = false;
	readFileInitLock.unlock();
	pagableBufferCond[pagableBufferIdx].notify_one();
	pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
	endReading = clock();
	timeReading += (endReading - startReading) * 1000 / CLOCKS_PER_SEC;
	//Read the rest
	while (true) {
		unique_lock<mutex> readFileIterLock(pagableBufferMutex[pagableBufferIdx]);
		while (pagableBufferObsolete[pagableBufferIdx] == false) {
			pagableBufferCond[pagableBufferIdx].wait(readFileIterLock);
		}
		startReading = clock();

		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);

		if (charArrayBuffer.GetLen() == 0) {
			readFileIterLock.unlock();
			pagableBufferCond[pagableBufferIdx].notify_all();
			break;	//Read nothing
		}
		++count;
		memcpy(pagableBuffer[pagableBufferIdx], 
			   overlap, 
			   WINDOW_SIZE - 1);		//copy the overlap into current part
		memcpy(&pagableBuffer[pagableBufferIdx][WINDOW_SIZE - 1], 
			   charArrayBuffer.GetArr(), 
			   charArrayBuffer.GetLen());
		pagableBufferLen[pagableBufferIdx] = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
		totalFileLen += charArrayBuffer.GetLen();
		pagableBufferObsolete[pagableBufferIdx] = false;
		memcpy(overlap, 
			   &pagableBuffer[pagableBufferIdx][charArrayBuffer.GetLen()], 
			   WINDOW_SIZE - 1);	//copy the last window into overlap
		readFileIterLock.unlock();
		pagableBufferCond[pagableBufferIdx].notify_one();
		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		endReading = clock();
		timeReading += (endReading - startReading) * 1000 / CLOCKS_PER_SEC;
	}
	IO::Print("File size: %s\n", IO::InterpretSize(totalFileLen));
	unique_lock<mutex> readFileEndLock(readFileEndMutex);
	IO::Print("Need %d pagable buffers\n", count);
	readFileEnd = true;
	//In case the other threads stuck in waiting for condition variable
	pagableBufferCond[pagableBufferIdx].notify_all();

}

/*
* Transfer data from pagable buffer to pinned buffer.
* Because memory transfer between GPU device memory and
* pinned buffer is way faster than between pagable buffer.
*/
//void Transfer() {
//	int pagableBufferIdx = 0;
//	int fixedBufferIdx = 0;
//	while (true) {
//		//Get pagable buffer ready
//		unique_lock<mutex> pagableLock(pagableBufferMutex[pagableBufferIdx]);
//		while (pagableBufferObsolete[pagableBufferIdx] == true) {
//			cout << 2 << endl;
//			if (readFileEnd) {
//				transfer_end = true;
//				cout << "end transfer \n";
//				return;
//			}
//			pagableBufferCond[pagableBufferIdx].wait(pagableLock);
//		}
//		//Get fixed buffer ready
//		unique_lock<mutex> fixedLock(fixed_buffer_mutex[fixedBufferIdx]);
//		while (fixed_buffer_obsolete[fixedBufferIdx] == false) {
//			cout << 3 << endl;
//			fixed_buffer_cond[fixedBufferIdx].wait(fixedLock);
//		}
//		start_t = clock();
//		fixedBufferLen[fixedBufferIdx] = pagableBufferLen[pagableBufferIdx];
//		memcpy(fixedBuffer[fixedBufferIdx], 
//			   pagableBuffer[pagableBufferIdx], 
//			   fixedBufferLen[fixedBufferIdx]);
//		//pagable buffer is still not obsolete here!
//		fixed_buffer_obsolete[fixedBufferIdx] = false;
//		pagableLock.unlock();
//		pagableBufferCond[pagableBufferIdx].notify_one();
//		fixedLock.unlock();
//		fixed_buffer_cond[fixedBufferIdx].notify_one();
//		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
//		fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
//		end_t = clock();
//		time_t += (end_t - start_t) * 1000 / CLOCKS_PER_SEC;
//	}
//}

/*
* Call the GPU kernel function to compute the Rabin hash value
* of each sliding window
*/
void Harens::ChunkingKernel() {
	int pagableBufferIdx = 0;
	int fixedBufferIdx = 0;
	int streamIdx = 0;
	while (true) {
		//Wait for the last process of this stream to finish
		//Get pagable buffer ready
		unique_lock<mutex> pagableLock(pagableBufferMutex[pagableBufferIdx]);
		while (pagableBufferObsolete[pagableBufferIdx] == true) {
			unique_lock<mutex> readFileEndLock(readFileEndMutex);
			if (readFileEnd) {
				unique_lock<mutex> chunkingKernelEndLock(chunkingKernelEndMutex);
				chunkingKernelEnd = true;
				return;
			}
			readFileEndLock.unlock();
			pagableBufferCond[pagableBufferIdx].wait(pagableLock);
		}

		//Get result host ready
		unique_lock<mutex> resultHostLock(hostResultMutex[fixedBufferIdx]);
		while (hostResultExecuting[fixedBufferIdx] == true) {
			hostResultCond[fixedBufferIdx].wait(resultHostLock);
		}

		startChunkingKernel = clock();
		fixedBufferLen[fixedBufferIdx] = pagableBufferLen[pagableBufferIdx];
		memcpy(fixedBuffer[fixedBufferIdx], 
			   pagableBuffer[pagableBufferIdx], 
			   fixedBufferLen[fixedBufferIdx]);
		//pagable buffer is still not obsolete here!
		pagableLock.unlock();
		pagableBufferCond[pagableBufferIdx].notify_one();
		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;

		re.RabinHashAsync(kernelInputBuffer[fixedBufferIdx], 
						  fixedBuffer[fixedBufferIdx], 
						  fixedBufferLen[fixedBufferIdx],
						  kernelResultBuffer[fixedBufferIdx], 
						  hostResultBuffer[fixedBufferIdx],
						  stream[streamIdx]);

		hostResultLen[fixedBufferIdx] = fixedBufferLen[fixedBufferIdx] - WINDOW_SIZE + 1;
		hostResultExecuting[fixedBufferIdx] = true;
		resultHostLock.unlock();
		hostResultCond[fixedBufferIdx].notify_one();
		fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
		streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
		endChunkingKernel = clock();
		timeChunkingKernel += (endChunkingKernel - startChunkingKernel) * 1000 / CLOCKS_PER_SEC;
	}
}

/*
* Mark the beginning of a window as a fingerprint based on the MODP rule.
* The fingerprints divide the stream into chunks.
*/
void Harens::ChunkingResultProc() {
	int resultHostIdx = 0;
	int streamIdx = 0;
		
	while (true) {
		//Get result host ready
		unique_lock<mutex> resultHostLock(hostResultMutex[resultHostIdx]);
		while (hostResultExecuting[resultHostIdx] == false) {
			unique_lock<mutex> chunkingKernelEndLock(chunkingKernelEndMutex);
			if (chunkingKernelEnd) {
				unique_lock<mutex> chukingProcEndLock(chunkingResultProcEndMutex);
				chunkingResultProcEnd = true;
				return;
			}
			chunkingKernelEndLock.unlock();
			hostResultCond[resultHostIdx].wait(resultHostLock);
		}
		cudaStreamSynchronize(stream[streamIdx]);
		//Get the chunking result ready
		unique_lock<mutex> chunkingResultLock(chunkingResultMutex[streamIdx]);
		while (chunkingResultObsolete[streamIdx] == false) {
			chunkingResultCond[streamIdx].wait(chunkingResultLock);
		}
			
		startChunkPartitioning = clock();
		//all the inputs other than the last one contains #MAX_WINDOW_NUM of windows
		int chunkingResultIdx = 0;
		unsigned int resultHostLen = hostResultLen[resultHostIdx];
		for (unsigned int j = 0; j < resultHostLen; ++j) {
			if (hostResultBuffer[resultHostIdx][j] == 0) {
				chunkingResultBuffer[streamIdx][chunkingResultIdx++] = j;
			}
		}

		chunkingResultLen[streamIdx] = chunkingResultIdx;

		hostResultExecuting[resultHostIdx] = false;
		chunkingResultObsolete[streamIdx] = false;
		resultHostLock.unlock();
		hostResultCond[resultHostIdx].notify_one();
		chunkingResultLock.unlock();
		chunkingResultCond[streamIdx].notify_one();

		streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
		resultHostIdx = (resultHostIdx + 1) % FIXED_BUFFER_NUM;
		endChunkPartitioning = clock();
		timeChunkPartitioning += (endChunkPartitioning - startChunkPartitioning) * 1000 / CLOCKS_PER_SEC;
	}
}

/*
* Compute a non-collision hash (SHA-1) value for each chunk
* Chunks are divided into segments and processed by function ChunkSegmentHashing.
*/
void Harens::ChunkHashing() {
	int pagableBufferIdx = 0;
	int chunkingResultIdx = 0;
	while (true) {
		//Get the chunking result ready
		unique_lock<mutex> chunkingResultLock(chunkingResultMutex[chunkingResultIdx]);
		while (chunkingResultObsolete[chunkingResultIdx] == true) {
			unique_lock<mutex> chukingProcEndLock(chunkingResultProcEndMutex);
			if (chunkingResultProcEnd) {
				unique_lock<mutex> chunkHashingEndLock(chunkHashingEndMutex);
				chunkHashingEnd = true;
				return;
			}
			chukingProcEndLock.unlock();
			chunkingResultCond[chunkingResultIdx].wait(chunkingResultLock);
		}

		//Pagable buffer is ready because it's never released
		unique_lock<mutex> pagableLock(pagableBufferMutex[pagableBufferIdx]);

		startChunkHashing = clock();
		for (int i = 0; i < mapperNum; ++i) {
			segmentThreads[i] = thread(std::mem_fn(&Harens::ChunkSegmentHashing)
				, this, pagableBufferIdx, chunkingResultIdx, i);
		}

		for (int i = 0; i < mapperNum; ++i) {
			segmentThreads[i].join();
		}

		pagableBufferObsolete[pagableBufferIdx] = true;
		chunkingResultObsolete[chunkingResultIdx] = true;
		pagableLock.unlock();
		pagableBufferCond[pagableBufferIdx].notify_one();
		chunkingResultLock.unlock();
		chunkingResultCond[chunkingResultIdx].notify_one();

		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		chunkingResultIdx = (chunkingResultIdx + 1) % RESULT_BUFFER_NUM;
		endChunkHashing = clock();
		timeChunkHashing += (endChunkHashing - startChunkHashing) * 1000 / CLOCKS_PER_SEC;
	}
}

/*
* Compute a non-collision hash (SHA-1) value for each chunk in the segment
*/
void Harens::ChunkSegmentHashing(int pagableBufferIdx, int chunkingResultIdx, int segmentNum) {
	int listSize = chunkingResultLen[chunkingResultIdx];
	unsigned int* chunkingResultSeg = &chunkingResultBuffer[chunkingResultIdx]
													  [segmentNum * listSize / mapperNum];
	int segLen = listSize / mapperNum;
	if ((segmentNum + 1) * listSize / mapperNum > listSize)
		segLen = listSize - segmentNum * listSize / mapperNum;
	re.ChunkHashingAsync(chunkingResultSeg, segLen, pagableBuffer[pagableBufferIdx],
		chunkHashQueuePool);
	/*tuple<unsigned long long, unsigned int> chunkInfo;
	unsigned long long toBeDel;
	do {
		chunkInfo = chunk_hash_queue[chunkingResultIdx][segmentNum].Pop();
		if (hashPool.FindAndAdd(get<0>(chunkInfo), toBeDel)) {
			totalDuplicationSize += get<1>(chunkInfo);
		}
	} while (get<1>(chunkInfo) != -1);*/
}

/*
* Match the chunks by their hash values
*/
void Harens::ChunkMatch(int hashPoolIdx) {
	unsigned char* toBeDel = nullptr;
	while (true) {
		if (chunkHashQueuePool.IsEmpty(hashPoolIdx)) {
			unique_lock<mutex> chunkHashingEndLock(chunkHashingEndMutex);
			if (chunkHashingEnd)
				return;
			else {
				chunkHashingEndLock.unlock();
				this_thread::sleep_for(std::chrono::milliseconds(500));
				continue;
			}
		}
			
		startChunkMatching = clock();

		tuple<unsigned char*, unsigned int> valLenPair = chunkHashQueuePool.Pop(hashPoolIdx);
		if (circHashPool[hashPoolIdx].FindAndAdd(get<0>(valLenPair), toBeDel))
			duplicationSize[hashPoolIdx] += get<1>(valLenPair);
		if (toBeDel != nullptr) {
			//Remove chunk corresponding to toBeDel from storage
		}

		endChunkMatching = clock();
		timeChunkMatching += (endChunkMatching - startChunkMatching) * 1000 / CLOCKS_PER_SEC;
	}
}