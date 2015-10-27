#include "Harens.h"
using namespace std;

Harens::Harens(int mapperNum, int reducerNum) 
	: charArrayBuffer(MAX_BUFFER_LEN), chunk_hash_queue_pool(reducerNum) {
	this->mapperNum = mapperNum;
	this->reducerNum = reducerNum;
	segment_threads = new thread[mapperNum];
	chunk_match_threads = new thread[reducerNum];
	circ_hash_pool = new LRUHash[reducerNum];
	duplication_size = new unsigned int[reducerNum];
	for (int i = 0; i < reducerNum; ++i) {
		circ_hash_pool[i] = LRUHash(MAX_CHUNK_NUM / reducerNum);
		duplication_size[i] = 0;
	}

	re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::NonMultifingerprint);
	//initialize pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) {
		pagable_buffer[i] = new char[MAX_BUFFER_LEN];
		pagable_buffer_obsolete[i] = true;
	}
	//initialize fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaMallocHost((void**)&fixed_buffer[i], MAX_BUFFER_LEN);
		//fixed_buffer_obsolete[i] = true;
	}
	//initialize chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaMalloc((void**)&input_kernel[i], MAX_BUFFER_LEN);
		cudaMalloc((void**)&result_kernel[i], MAX_WINDOW_NUM * BYTES_IN_ULONG);
		cudaMallocHost((void**)&result_host[i], MAX_WINDOW_NUM * BYTES_IN_ULONG);
		result_host_obsolete[i] = true;
		result_host_executing[i] = false;
	}
	//initialize chunking result processing
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) {
		cudaStreamCreate(&stream[i]);
		chunking_result[i] = new unsigned int[MAX_WINDOW_NUM];
		chunking_result_obsolete[i] = true;
	}
	//initialize chunk hashing
	//for (int i = 0; i < STREAM_NUM; ++i) {
	//	segment_threads[i] = new thread[FINGERPRINTING_THREAD_NUM];
	//	for (int j = 0; j < FINGERPRINTING_THREAD_NUM; ++j) {
	//		//MAX_WINDOW_NUM / 4 is a guess of the upper bound of the number of chunks
	//		/*chunk_hashing_value_queue[i][j] = LRUUcharArrayQueue(MAX_WINDOW_NUM / 4);
	//		chunk_len_queue[i][j] = LRUUintQueue(MAX_WINDOW_NUM / 4);*/
	//	}
	//}
}

Harens::~Harens() {
	delete[] segment_threads;
	delete[] chunk_match_threads;
	delete[] circ_hash_pool;
	delete[] duplication_size;

	//destruct chunk hashing & matching
	/*for (int i = 0; i < STREAM_NUM; ++i) {
	delete[] segment_threads[i];
	}*/
	//destruct chunking result proc
	for (int i = 0; i < RESULT_BUFFER_NUM; ++i) {
		cudaStreamDestroy(stream[i]);
		delete[] chunking_result[i];
	}
	//destruct chunking kernel ascychronize
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaFree(input_kernel[i]);
		cudaFree(result_kernel[i]);
		cudaFreeHost(result_host[i]);
	}
	//destruct fixed buffer
	for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
		cudaFreeHost(fixed_buffer[i]);
	}
	//destruct pagable buffer
	for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) {
		delete[] pagable_buffer[i];
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
		chunk_match_threads[i] = thread(std::mem_fn(&Harens::ChunkMatch), this, i);

	//tTransfer.join();
	tChunkingKernel.join();
	tChunkingResultProc.join();
	tChunkHashing.join();
	for (int i = 0; i < reducerNum; ++i)
		chunk_match_threads[i].join();
	//tRoundQuery.join();

	end = clock();
	time = (end - start) * 1000 / CLOCKS_PER_SEC;
	IO::Print("Read file time: %f ms\n", time_r);
	//printf("Transfer time: %f ms\n", time_t);
	IO::Print("Chunking kernel time: %f ms\n", time_ck);
	IO::Print("Chunking processing time: %f ms\n", time_cp);
	IO::Print("Map (Chunk hashing) time: %f ms\n", time_ch);
	IO::Print("Reduce (Chunk matching) time %f ms\n", time_cm);
	IO::Print("Total time: %f ms\n", time);
	for (int i = 0; i < reducerNum; ++i)
		total_duplication_size += duplication_size[i];
	IO::Print("Found %s of redundancy, which is %f %% of file\n"
		, IO::InterpretSize(total_duplication_size)
		, total_duplication_size * 100.0 / file_length);

	return 0;
}

void Harens::ReadFile() {
	int pagableBufferIdx = 0;
	unsigned int curFilePos = 0;
	int curWindowNum;
	PcapReader fileReader;
	//Read the first part
	unique_lock<mutex> readFileInitLock(pagable_buffer_mutex[pagableBufferIdx]);
	start_r = clock();
	if (IO::FILE_FORMAT == PlainText) {
		ifs = ifstream(IO::input_file_name, ios::in | ios::binary | ios::ate);
		if (!ifs.is_open()) {
			printf("Can not open file %s\n", IO::input_file_name);
			return;
		}
		file_length = ifs.tellg();
		ifs.seekg(0, ifs.beg);
		IO::Print("File size: %s\n", IO::InterpretSize(file_length));
		pagable_buffer_len[pagableBufferIdx] = min(MAX_BUFFER_LEN, file_length - curFilePos);
		curWindowNum = pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1;
		ifs.read(pagable_buffer[pagableBufferIdx], pagable_buffer_len[pagableBufferIdx]);
		curFilePos += curWindowNum;
	}
	else if (IO::FILE_FORMAT == Pcap) {
		fileReader.SetupPcapHandle(IO::input_file_name);
		fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN);
		pagable_buffer_len[pagableBufferIdx] = charArrayBuffer.GetLen();
		memcpy(pagable_buffer[pagableBufferIdx], charArrayBuffer.GetArr(), pagable_buffer_len[pagableBufferIdx]);
		file_length += pagable_buffer_len[pagableBufferIdx];
	}
	else
		fprintf(stderr, "Unknown file format\n");
	++count;

	memcpy(overlap, &pagable_buffer[pagableBufferIdx][pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
	pagable_buffer_obsolete[pagableBufferIdx] = false;
	readFileInitLock.unlock();
	pagable_buffer_cond[pagableBufferIdx].notify_one();
	pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
	end_r = clock();
	time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
	//Read the rest
	if (IO::FILE_FORMAT == PlainText) {
		//curWindowNum will be less than MAX_BUFFER_LEN - WINDOW_SIZE + 1 when reading the last part
		while (curWindowNum == MAX_WINDOW_NUM) {
			unique_lock<mutex> readFileIterLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == false) {
				pagable_buffer_cond[pagableBufferIdx].wait(readFileIterLock);
			}
			++count;
			start_r = clock();
			pagable_buffer_len[pagableBufferIdx] = min(MAX_BUFFER_LEN, file_length - curFilePos);
			curWindowNum = pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1;
			memcpy(pagable_buffer[pagableBufferIdx], overlap, WINDOW_SIZE - 1);		//copy the overlap into current part
			ifs.read(&pagable_buffer[pagableBufferIdx][WINDOW_SIZE - 1], curWindowNum);
			memcpy(overlap, &pagable_buffer[pagableBufferIdx][curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
			pagable_buffer_obsolete[pagableBufferIdx] = false;
			readFileIterLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			curFilePos += curWindowNum;
			end_r = clock();
			time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		}
		ifs.close();
	}
	else if (IO::FILE_FORMAT == Pcap) {
		while (true) {
			unique_lock<mutex> readFileIterLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == false) {
				pagable_buffer_cond[pagableBufferIdx].wait(readFileIterLock);
			}
			start_r = clock();
			fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);

			if (charArrayBuffer.GetLen() == 0) {
				readFileIterLock.unlock();
				pagable_buffer_cond[pagableBufferIdx].notify_all();
				break;	//Read nothing
			}
			++count;
			memcpy(pagable_buffer[pagableBufferIdx], overlap, WINDOW_SIZE - 1);		//copy the overlap into current part
			memcpy(&pagable_buffer[pagableBufferIdx][WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
			pagable_buffer_len[pagableBufferIdx] = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
			file_length += charArrayBuffer.GetLen();
			pagable_buffer_obsolete[pagableBufferIdx] = false;
			memcpy(overlap, &pagable_buffer[pagableBufferIdx][charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
			readFileIterLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			end_r = clock();
			time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		}
		IO::Print("File size: %s\n", IO::InterpretSize(file_length));
	}
	else
		fprintf(stderr, "Unknown file format\n");
	unique_lock<mutex> readFileEndLock(read_file_end_mutex);
	IO::Print("Need %d pagable buffers\n", count);
	read_file_end = true;
	//In case the other threads stuck in waiting for condition variable
	pagable_buffer_cond[pagableBufferIdx].notify_all();

}

//void Transfer() {
//	int pagableBufferIdx = 0;
//	int fixedBufferIdx = 0;
//	while (true) {
//		//Get pagable buffer ready
//		unique_lock<mutex> pagableLock(pagable_buffer_mutex[pagableBufferIdx]);
//		while (pagable_buffer_obsolete[pagableBufferIdx] == true) {
//			cout << 2 << endl;
//			if (read_file_end) {
//				transfer_end = true;
//				cout << "end transfer \n";
//				return;
//			}
//			pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
//		}
//		//Get fixed buffer ready
//		unique_lock<mutex> fixedLock(fixed_buffer_mutex[fixedBufferIdx]);
//		while (fixed_buffer_obsolete[fixedBufferIdx] == false) {
//			cout << 3 << endl;
//			fixed_buffer_cond[fixedBufferIdx].wait(fixedLock);
//		}
//		start_t = clock();
//		fixed_buffer_len[fixedBufferIdx] = pagable_buffer_len[pagableBufferIdx];
//		memcpy(fixed_buffer[fixedBufferIdx], pagable_buffer[pagableBufferIdx], fixed_buffer_len[fixedBufferIdx]);
//		//pagable buffer is still not obsolete here!
//		fixed_buffer_obsolete[fixedBufferIdx] = false;
//		pagableLock.unlock();
//		pagable_buffer_cond[pagableBufferIdx].notify_one();
//		fixedLock.unlock();
//		fixed_buffer_cond[fixedBufferIdx].notify_one();
//		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
//		fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
//		end_t = clock();
//		time_t += (end_t - start_t) * 1000 / CLOCKS_PER_SEC;
//	}
//}

void Harens::ChunkingKernel() {
	int pagableBufferIdx = 0;
	int fixedBufferIdx = 0;
	int streamIdx = 0;
	while (true) {
		//Wait for the last process of this stream to finish
		//Get pagable buffer ready
		unique_lock<mutex> pagableLock(pagable_buffer_mutex[pagableBufferIdx]);
		while (pagable_buffer_obsolete[pagableBufferIdx] == true) {
			unique_lock<mutex> readFileEndLock(read_file_end_mutex);
			if (read_file_end) {
				unique_lock<mutex> chunkingKernelEndLock(chunking_kernel_end_mutex);
				chunking_kernel_end = true;
				return;
			}
			readFileEndLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
		}

		//Get result host ready
		unique_lock<mutex> resultHostLock(result_host_mutex[fixedBufferIdx]);
		while (result_host_executing[fixedBufferIdx] == true) {
			result_host_cond[fixedBufferIdx].wait(resultHostLock);
		}

		start_ck = clock();
		fixed_buffer_len[fixedBufferIdx] = pagable_buffer_len[pagableBufferIdx];
		memcpy(fixed_buffer[fixedBufferIdx], pagable_buffer[pagableBufferIdx], fixed_buffer_len[fixedBufferIdx]);
		//pagable buffer is still not obsolete here!
		pagableLock.unlock();
		pagable_buffer_cond[pagableBufferIdx].notify_one();
		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;

		re.RabinHashAsync(input_kernel[fixedBufferIdx], fixed_buffer[fixedBufferIdx], fixed_buffer_len[fixedBufferIdx],
			result_kernel[fixedBufferIdx], result_host[fixedBufferIdx], stream[streamIdx]);

		result_host_len[fixedBufferIdx] = fixed_buffer_len[fixedBufferIdx] - WINDOW_SIZE + 1;
		result_host_executing[fixedBufferIdx] = true;
		resultHostLock.unlock();
		result_host_cond[fixedBufferIdx].notify_one();
		fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
		streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
		end_ck = clock();
		time_ck += (end_ck - start_ck) * 1000 / CLOCKS_PER_SEC;
	}
}

void Harens::ChunkingResultProc() {
	int resultHostIdx = 0;
	int streamIdx = 0;
		
	while (true) {
		//Get result host ready
		unique_lock<mutex> resultHostLock(result_host_mutex[resultHostIdx]);
		while (result_host_executing[resultHostIdx] == false) {
			unique_lock<mutex> chunkingKernelEndLock(chunking_kernel_end_mutex);
			if (chunking_kernel_end) {
				unique_lock<mutex> chukingProcEndLock(chunking_proc_end_mutex);
				chunking_proc_end = true;
				return;
			}
			chunkingKernelEndLock.unlock();
			result_host_cond[resultHostIdx].wait(resultHostLock);
		}
		cudaStreamSynchronize(stream[streamIdx]);
		//Get the chunking result ready
		unique_lock<mutex> chunkingResultLock(chunking_result_mutex[streamIdx]);
		while (chunking_result_obsolete[streamIdx] == false) {
			chunking_result_cond[streamIdx].wait(chunkingResultLock);
		}
			
		start_cp = clock();
		//all the inputs other than the last one contains #MAX_WINDOW_NUM of windows
		int chunkingResultIdx = 0;
		unsigned int resultHostLen = result_host_len[resultHostIdx];
		for (unsigned int j = 0; j < resultHostLen; ++j) {
			if ((result_host[resultHostIdx][j] & P_MINUS) == 0) {
				chunking_result[streamIdx][chunkingResultIdx++] = j;
			}
		}

		chunking_result_len[streamIdx] = chunkingResultIdx;

		result_host_executing[resultHostIdx] = false;
		chunking_result_obsolete[streamIdx] = false;
		resultHostLock.unlock();
		result_host_cond[resultHostIdx].notify_one();
		chunkingResultLock.unlock();
		chunking_result_cond[streamIdx].notify_one();

		streamIdx = (streamIdx + 1) % RESULT_BUFFER_NUM;
		resultHostIdx = (resultHostIdx + 1) % FIXED_BUFFER_NUM;
		end_cp = clock();
		time_cp += (end_cp - start_cp) * 1000 / CLOCKS_PER_SEC;
	}
}

void Harens::ChunkHashing() {
	int pagableBufferIdx = 0;
	int chunkingResultIdx = 0;
	while (true) {
		//Get pagable buffer ready
		unique_lock<mutex> pagableLock(pagable_buffer_mutex[pagableBufferIdx]);
		while (pagable_buffer_obsolete[pagableBufferIdx] == true) {
			unique_lock<mutex> chukingProcEndLock(chunking_proc_end_mutex);
			if (chunking_proc_end) {
				unique_lock<mutex> chunkHashingEndLock(chunk_hashing_end_mutex);
				chunk_hashing_end = true;
				return;
			}
			chukingProcEndLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
		}
		//Get the chunking result ready
		unique_lock<mutex> chunkingResultLock(chunking_result_mutex[chunkingResultIdx]);
		while (chunking_result_obsolete[chunkingResultIdx] == true) {
			unique_lock<mutex> chukingProcEndLock(chunking_proc_end_mutex);
			if (chunking_proc_end) {
				unique_lock<mutex> chunkHashingEndLock(chunk_hashing_end_mutex);
				chunk_hashing_end = true;
				return;
			}
			chukingProcEndLock.unlock();
			chunking_result_cond[chunkingResultIdx].wait(chunkingResultLock);
		}

		start_ch = clock();
		for (int i = 0; i < mapperNum; ++i) {
			segment_threads[i] = thread(std::mem_fn(&Harens::ChunkSegmentHashing)
				, this, pagableBufferIdx, chunkingResultIdx, i);
		}

		for (int i = 0; i < mapperNum; ++i) {
			segment_threads[i].join();
		}

		pagable_buffer_obsolete[pagableBufferIdx] = true;
		chunking_result_obsolete[chunkingResultIdx] = true;
		pagableLock.unlock();
		pagable_buffer_cond[pagableBufferIdx].notify_one();
		chunkingResultLock.unlock();
		chunking_result_cond[chunkingResultIdx].notify_one();

		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		chunkingResultIdx = (chunkingResultIdx + 1) % RESULT_BUFFER_NUM;
		end_ch = clock();
		time_ch += (end_ch - start_ch) * 1000 / CLOCKS_PER_SEC;
	}
}

void Harens::ChunkSegmentHashing(int pagableBufferIdx, int chunkingResultIdx, int segmentNum) {
	int listSize = chunking_result_len[chunkingResultIdx];
	unsigned int* chunkingResultSeg = &chunking_result[chunkingResultIdx][segmentNum * listSize / mapperNum];
	int segLen = listSize / mapperNum;
	if ((segmentNum + 1) * listSize / mapperNum > listSize)
		segLen = listSize - segmentNum * listSize / mapperNum;
	re.ChunkHashingAsync(chunkingResultSeg, segLen, pagable_buffer[pagableBufferIdx],
		chunk_hash_queue_pool);
	/*tuple<unsigned long long, unsigned int> chunkInfo;
	unsigned long long toBeDel;
	do {
		chunkInfo = chunk_hash_queue[chunkingResultIdx][segmentNum].Pop();
		if (hash_pool.FindAndAdd(get<0>(chunkInfo), toBeDel)) {
			total_duplication_size += get<1>(chunkInfo);
		}
	} while (get<1>(chunkInfo) != -1);*/
}

void Harens::ChunkMatch(int hashPoolIdx) {
	unsigned char* toBeDel = nullptr;
	while (true) {
		if (chunk_hash_queue_pool.IsEmpty(hashPoolIdx)) {
			unique_lock<mutex> chunkHashingEndLock(chunk_hashing_end_mutex);
			if (chunk_hashing_end)
				return;
			else {
				chunkHashingEndLock.unlock();
				this_thread::sleep_for(std::chrono::milliseconds(500));
				continue;
			}
		}
			
		start_cm = clock();

		tuple<unsigned char*, unsigned int> valLenPair = chunk_hash_queue_pool.Pop(hashPoolIdx);
		if (circ_hash_pool[hashPoolIdx].FindAndAdd(get<0>(valLenPair), toBeDel))
			duplication_size[hashPoolIdx] += get<1>(valLenPair);
		if (toBeDel != nullptr) {
			//Remove chunk corresponding to toBeDel from storage
		}

		end_cm = clock();
		time_cm += (end_cm - start_cm) * 1000 / CLOCKS_PER_SEC;
	}
}