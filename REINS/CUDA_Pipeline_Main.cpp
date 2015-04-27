#include "CUDA_Pipeline_Main.h"
using namespace std;

namespace CUDA_Pipeline_Namespace {
	//constants
	const uint MAX_BUFFER_LEN = MAX_KERNEL_INPUT_LEN;
	const uint MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;
	const int PAGABLE_BUFFER_NUM = 10;
	const int FIXED_BUFFER_NUM = 4;
	const int STREAM_NUM = 3;
	const int FINGERPRINTING_THREAD_NUM = 6;
	//determin if one thread is end
	bool read_file_end = false;
	bool transfer_end = false;
	bool chunking_kernel_end = false;
	bool chunking_proc_end = false;
	bool chunk_hashing_end = false;
	//file
	ifstream fin;
	uint file_len;
	char overlap[WINDOW_SIZE - 1];
	//pagable buffer
	array<char*, PAGABLE_BUFFER_NUM> pagable_buffer;
	array<uint, PAGABLE_BUFFER_NUM> pagable_buffer_len;
	array<mutex, PAGABLE_BUFFER_NUM> pagable_buffer_mutex;
	array<condition_variable, PAGABLE_BUFFER_NUM> pagable_buffer_cond;
	array<bool, PAGABLE_BUFFER_NUM> pagable_buffer_obsolete;
	//fixed buffer
	array<char*, FIXED_BUFFER_NUM> fixed_buffer;
	array<uint, FIXED_BUFFER_NUM> fixed_buffer_len;
	array<mutex, FIXED_BUFFER_NUM> fixed_buffer_mutex;
	array<condition_variable, FIXED_BUFFER_NUM> fixed_buffer_cond;
	array<bool, FIXED_BUFFER_NUM> fixed_buffer_obsolete;
	//RedundancyEliminator_CUDA
	RedundancyEliminator_CUDA re;
	//chunking kernel asynchronize
	array<char*, FIXED_BUFFER_NUM> input_kernel;
	array<ulong*, FIXED_BUFFER_NUM> result_kernel;
	array<ulong*, FIXED_BUFFER_NUM> result_host;
	array<uint, FIXED_BUFFER_NUM> result_host_len;
	array<mutex, FIXED_BUFFER_NUM> result_host_mutex;
	array<condition_variable, FIXED_BUFFER_NUM> result_host_cond;
	array<bool, FIXED_BUFFER_NUM> result_host_obsolete;
	//chunking result processing
	array<cudaStream_t, STREAM_NUM> stream;
	array<uint*, STREAM_NUM> chunking_result;
	array<uint, STREAM_NUM> chunking_result_len;
	array<mutex, STREAM_NUM> chunking_result_mutex;
	array<condition_variable, STREAM_NUM> chunking_result_cond;
	array<bool, STREAM_NUM> chunking_result_obsolete;
	//chunk hashing
	array<thread*, STREAM_NUM> segment_threads;
	array<uchar**, STREAM_NUM> chunk_hashing_value_list;
	array<uint**, STREAM_NUM> chunk_len_list;	//This is only for simulation, in real case don't need the "chunk length"
	array<mutex*, STREAM_NUM> chunk_hash_mutex;
	//chunk matching 
	uint total_duplication_size = 0;
	//Time
	clock_t start, end, start_r, end_r, start_t, end_t, start_ck, end_ck, start_cp, end_cp, start_ch, end_ch, start_cm, end_cm;
	double time = 0, time_r = 0, time_t = 0, time_ck = 0, time_cp = 0, time_ch, time_cm;
	
	int CUDA_Pipeline_Main(int argc, char* argv[]) {
		cout << "\n============ CUDA Implementation With Pipeline and Circular Queue ============\n";
		if (argc != 2) {
			printf("Usage: %s <filename>\n", argv[0]);
			system("pause");
			return -1;
		}

		//Initialize file reader
		fin = ifstream(argv[1], ios::in | ios::binary | ios::ate);
		if (!fin.is_open()) {
			cout << "Can not open file " << argv[1] << endl;
			system("pause");
			return -1;
		}
		file_len = fin.tellg();
		fin.seekg(0, fin.beg);
		cout << "File size: " << file_len / 1024 << " KB\n";

		re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::MultiFingerprint);
		//initialize pagable buffer
		for (int i = 0; i < PAGABLE_BUFFER_NUM; ++i) {
			pagable_buffer[i] = new char[MAX_BUFFER_LEN];
			pagable_buffer_obsolete[i] = true;
		}
		//initialize fixed buffer
		for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
			cudaMallocHost((void**)&fixed_buffer[i], MAX_BUFFER_LEN);
			fixed_buffer_obsolete[i] = true;
		}
		//initialize chunking kernel ascychronize
		for (int i = 0; i < FIXED_BUFFER_NUM; ++i) {
			cudaMalloc((void**)&input_kernel[i], MAX_BUFFER_LEN);
			cudaMalloc((void**)&result_kernel[i], MAX_WINDOW_NUM * BYTES_IN_ULONG);
			cudaMallocHost((void**)&result_host[i], MAX_WINDOW_NUM * BYTES_IN_ULONG);
			result_host_obsolete[i] = true;
		}
		//initialize chunking result processing
		for (int i = 0; i < STREAM_NUM; ++i) {
			cudaStreamCreate(&stream[i]);
			chunking_result[i] = new uint[MAX_WINDOW_NUM];
			chunking_result_obsolete[i] = true;
		}
		//initialize chunk hashing
		for (int i = 0; i < STREAM_NUM; ++i) {
			segment_threads[i] = new thread[FINGERPRINTING_THREAD_NUM];
			chunk_hashing_value_list[i] = new uchar*[FINGERPRINTING_THREAD_NUM];
			chunk_len_list[i] = new uint*[FINGERPRINTING_THREAD_NUM];
			chunk_hash_mutex[i] = new mutex[FINGERPRINTING_THREAD_NUM];
			for (int j = 0; j < FINGERPRINTING_THREAD_NUM; ++j) {
				//MAX_WINDOW_NUM / 4 is a guess of the upper bound of the number of chunks
				chunk_hashing_value_list[i][j] = new uchar[MAX_WINDOW_NUM / 4 * SHA256_DIGEST_LENGTH];
				chunk_len_list[i][j] = new uint[MAX_WINDOW_NUM / 4];
			}
		}

		start = clock();

		//Create threads
		thread tReadFile(ReadFile);
		thread tTransfer(Transfer);
		thread tChunkingKernel(ChunkingKernel);
		thread tChunkingResultProc(ChunkingResultProc);
		thread tChunkHashing;
		tChunkHashing = thread(ChunkHashing);

		tReadFile.join();
		tTransfer.join();
		tChunkingKernel.join();
		tChunkingResultProc.join();
		tChunkHashing.join();

		end = clock();
		time = (end - start) * 1000 / CLOCKS_PER_SEC;
		
		printf("Read file time: %f ms\n", time_r);
		printf("Transfer time: %f ms\n", time_t);
		printf("Chunking kernel time: %f ms\n", time_ck);
		printf("Chunking processing time: %f ms\n", time_cp);
		printf("Chunk hashing time: %f ms\n", time_ch);
		printf("Round query chunk matching time %f ms\n", time_cm);
		printf("Total time: %f ms\n", time);
		printf("Found %d bytes of redundancy, which is %f percent of file\n", total_duplication_size, total_duplication_size * 100.0 / file_len);

		//destruct chunk hashing & matching
		for (int i = 0; i < STREAM_NUM; ++i) {
			for (int j = 0; j < FINGERPRINTING_THREAD_NUM; ++j) {
				delete[] chunk_hashing_value_list[i][j];
				delete[] chunk_len_list[i][j];
			}
			delete[] segment_threads[i];
			delete[] chunk_hashing_value_list[i];
			delete[] chunk_len_list[i];
			delete[] chunk_hash_mutex[i];
		}
		//destruct chunking result proc
		for (int i = 0; i < STREAM_NUM; ++i) {
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
		return 0;
	}

	void ReadFile() {
		int pagableBufferIdx = 0;
		uint curFilePos = 0;
		int curWindowNum;
		//Read the first part
		unique_lock<mutex> readFileInitLock(pagable_buffer_mutex[pagableBufferIdx]);
		start_r = clock();
		pagable_buffer_len[pagableBufferIdx] = min(MAX_BUFFER_LEN, file_len - curFilePos);
		curWindowNum = pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1;
		fin.read(pagable_buffer[pagableBufferIdx], pagable_buffer_len[pagableBufferIdx]);
		pagable_buffer_obsolete[pagableBufferIdx] = false;
		memcpy(overlap, &pagable_buffer[pagableBufferIdx][curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
		readFileInitLock.unlock();
		pagable_buffer_cond[pagableBufferIdx].notify_one();
		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		curFilePos += curWindowNum;
		end_r = clock();
		time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		//Read the rest
		//curWindowNum will be less than MAX_BUFFER_LEN - WINDOW_SIZE + 1 when reading the last part
		while (curWindowNum == MAX_WINDOW_NUM) {
			unique_lock<mutex> readFileIterLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == false) {
				pagable_buffer_cond[pagableBufferIdx].wait(readFileIterLock);
			}
			start_r = clock();
			pagable_buffer_len[pagableBufferIdx] = min(MAX_BUFFER_LEN, file_len - curFilePos);
			curWindowNum = pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1;
			memcpy(pagable_buffer[pagableBufferIdx], overlap, WINDOW_SIZE - 1);		//copy the overlap into current part
			fin.read(&pagable_buffer[pagableBufferIdx][WINDOW_SIZE - 1], curWindowNum);
			memcpy(overlap, &pagable_buffer[pagableBufferIdx][curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
			pagable_buffer_obsolete[pagableBufferIdx] = false;
			readFileIterLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			curFilePos += curWindowNum;
			end_r = clock();
			time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		}
		read_file_end = true;
		fin.close();
	}

	void Transfer() {
		int pagableBufferIdx = 0;
		int fixedBufferIdx = 0;
		while (true) {
			//Get pagable buffer ready
			unique_lock<mutex> pagableLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == true) {
				if (read_file_end) {
					transfer_end = true;
					return;
				}
				pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
			}
			//Get fixed buffer ready
			unique_lock<mutex> fixedLock(fixed_buffer_mutex[fixedBufferIdx]);
			while (fixed_buffer_obsolete[fixedBufferIdx] == false) {
				fixed_buffer_cond[fixedBufferIdx].wait(fixedLock);
			}
			start_t = clock();
			fixed_buffer_len[fixedBufferIdx] = pagable_buffer_len[pagableBufferIdx];
			memcpy(fixed_buffer[fixedBufferIdx], pagable_buffer[pagableBufferIdx], fixed_buffer_len[fixedBufferIdx]);
			//pagable buffer is still not obsolete here!
			fixed_buffer_obsolete[fixedBufferIdx] = false;
			pagableLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			fixedLock.unlock();
			fixed_buffer_cond[fixedBufferIdx].notify_one();
			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
			end_t = clock();
			time_t += (end_t - start_t) * 1000 / CLOCKS_PER_SEC;
		}
	}

	void ChunkingKernel() {
		int fixedBufferIdx = 0;
		int streamIdx = 0;
		while (true) {
			//Get fixed buffer ready
			unique_lock<mutex> fixedLock(fixed_buffer_mutex[fixedBufferIdx]);
			while (fixed_buffer_obsolete[fixedBufferIdx] == true) {
				if (transfer_end) {
					chunking_kernel_end = true;
					return;
				}
				fixed_buffer_cond[fixedBufferIdx].wait(fixedLock);
			}
			//Get result host ready
			unique_lock<mutex> resultHostLock(result_host_mutex[fixedBufferIdx]);
			while (result_host_obsolete[fixedBufferIdx] == false) {
				result_host_cond[fixedBufferIdx].wait(resultHostLock);
			}
			start_ck = clock();
			re.RabinHashAsync(input_kernel[fixedBufferIdx], fixed_buffer[fixedBufferIdx], fixed_buffer_len[fixedBufferIdx],
				result_kernel[fixedBufferIdx], result_host[fixedBufferIdx], stream[streamIdx]);
			
			result_host_len[fixedBufferIdx] = fixed_buffer_len[fixedBufferIdx] - WINDOW_SIZE + 1;
			fixed_buffer_obsolete[fixedBufferIdx] = true;
			result_host_obsolete[fixedBufferIdx] = false;
			fixedLock.unlock();
			fixed_buffer_cond[fixedBufferIdx].notify_one();
			resultHostLock.unlock();
			result_host_cond[fixedBufferIdx].notify_one();
			fixedBufferIdx = (fixedBufferIdx + 1) % FIXED_BUFFER_NUM;
			streamIdx = (streamIdx + 1) % STREAM_NUM;
			end_ck = clock();
			time_ck += (end_ck - start_ck) * 1000 / CLOCKS_PER_SEC;
		}
	}

	void ChunkingResultProc() {
		int resultHostIdx = 0;
		int streamIdx = 0;
		
		while (true) {
			//Get result host ready
			cudaStreamSynchronize(stream[streamIdx]);
			unique_lock<mutex> resultHostLock(result_host_mutex[resultHostIdx]);
			while (result_host_obsolete[resultHostIdx] == true) {
				if (chunking_kernel_end) {
					chunking_proc_end = true;
					return;
				}
				result_host_cond[resultHostIdx].wait(resultHostLock);
			}
			//Get the chunking result ready
			unique_lock<mutex> chunkingResultLock(chunking_result_mutex[streamIdx]);
			while (chunking_result_obsolete[streamIdx] == false) {
				chunking_result_cond[streamIdx].wait(chunkingResultLock);
			}
			
			start_cp = clock();
			//all the inputs other than the last one contains #MAX_WINDOW_NUM of windows
			int chunkingResultIdx = 0;
			uint resultHostLen = result_host_len[resultHostIdx];
			for (uint j = 0; j < resultHostLen; ++j) {
				if ((result_host[resultHostIdx][j] & P_MINUS) == 0) {
					chunking_result[streamIdx][chunkingResultIdx++] = j;
				}
			}
			chunking_result_len[streamIdx] = chunkingResultIdx;

			result_host_obsolete[resultHostIdx] = true;
			chunking_result_obsolete[streamIdx] = false;
			resultHostLock.unlock();
			result_host_cond[resultHostIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[streamIdx].notify_one();

			streamIdx = (streamIdx + 1) % STREAM_NUM;
			resultHostIdx = (resultHostIdx + 1) % FIXED_BUFFER_NUM;
			end_cp = clock();
			time_cp += (end_cp - start_cp) * 1000 / CLOCKS_PER_SEC;
		}
	}

	void ChunkHashing() {
		int pagableBufferIdx = 0;
		int chunkingResultIdx = 0;
		while (true) {
			//Get pagable buffer ready
			unique_lock<mutex> pagableLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == true) {
				if (chunking_proc_end)
					return;
				pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
			}
			//Get the chunking result ready
			unique_lock<mutex> chunkingResultLock(chunking_result_mutex[chunkingResultIdx]);
			while (chunking_result_obsolete[chunkingResultIdx] == true) {
				if (chunking_proc_end)
					return;
				chunking_result_cond[chunkingResultIdx].wait(chunkingResultLock);
			}

			start_ch = clock();
			for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
				segment_threads[chunkingResultIdx][i] = thread(ChunkSegmentHashing, chunkingResultIdx, i);
			}

			for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
				segment_threads[chunkingResultIdx][i].join();
			}

			pagable_buffer_obsolete[pagableBufferIdx] = true;
			chunking_result_obsolete[chunkingResultIdx] = true;
			pagableLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[chunkingResultIdx].notify_one();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			chunkingResultIdx = (chunkingResultIdx + 1) % STREAM_NUM;
			end_ch = clock();
			time_ch += (end_ch - start_ch) * 1000 / CLOCKS_PER_SEC;
		}
	}

	void ChunkSegmentHashing(int chunkingResultIdx, int segmentNum) {
		int listSize = chunking_result_len[chunkingResultIdx];
		uint* chunkingResultSeg = &chunking_result[chunkingResultIdx][segmentNum * listSize / FINGERPRINTING_THREAD_NUM];
		int segLen = listSize / FINGERPRINTING_THREAD_NUM;
		if ((segmentNum + 1) * listSize / FINGERPRINTING_THREAD_NUM > listSize)
			segLen = listSize - segmentNum * listSize / FINGERPRINTING_THREAD_NUM;
		re.ChunkHashingAscyn(chunkingResultSeg, segLen, pagable_buffer[chunkingResultIdx],
			chunk_hashing_value_list[chunkingResultIdx][segmentNum], 
			chunk_len_list[chunkingResultIdx][segmentNum], chunk_hash_mutex[chunkingResultIdx][segmentNum]);
	}

	//void RoundQuery() {
	//	bool noHashValueFound, noHashValueFoundInLoop;
	//	int chunkResultIdx = 0;
	//	tuple<uchar*, uint> empty = tuple<uchar*, uint>(new uchar(' '), -1);
	//	uchar* hashValue;
	//	uint chunkLen;
	//	noHashValueFound = true;
	//	while (true) {
	//		noHashValueFoundInLoop = true;
	//		while (true) {
	//			for (int segmentNum = 0; segmentNum < FINGERPRINTING_THREAD_NUM; ++segmentNum) {
	//				tuple<uchar*, uint> hashValueAndLen = empty;
	//				chunk_hash_mutex[chunkResultIdx][segmentNum].lock();
	//				if (!hash_value_queue[threadIdx][segmentNum].empty()) {
	//					hashValueAndLen = hash_value_queue[threadIdx][segmentNum].front();
	//					hash_value_queue[threadIdx][segmentNum].pop_front();
	//					noHashValueFoundInLoop = false;
	//				}
	//				hash_value_queue_mutex[threadIdx][segmentNum].unlock();

	//				hashValue = get<0>(hashValueAndLen);
	//				chunkLen = get<1>(hashValueAndLen);
	//				if (chunkLen != -1) {
	//					if (circ_hash.Find(hashValue)) {
	//						total_duplication_size += chunkLen;
	//					}
	//					else {
	//						uchar* to_be_del = circ_hash.Add(hashValue);
	//						if (to_be_del != NULL)
	//							delete[] to_be_del;
	//						//In real software we are supposed to deal with the chunk in disk
	//					}
	//				}
	//			}
	//		}
	//		if (noHashValueFound) {
	//			kill_ascyn_matching_mutex.lock();
	//			if (kill_ascyn_matching) {
	//				kill_ascyn_matching_mutex.unlock();
	//				return;
	//			}
	//			kill_ascyn_matching_mutex.unlock();
	//			this_thread::sleep_for(chrono::microseconds(500));
	//		}
	//	}
	//}
}