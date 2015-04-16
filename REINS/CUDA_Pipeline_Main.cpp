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
	RedundancyEliminator_CUDA re(RedundancyEliminator_CUDA::MultiFingerprint);
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
	array<deque<uint>, STREAM_NUM> chunking_result;
	array<mutex, STREAM_NUM> chunking_result_mutex;
	array<condition_variable, STREAM_NUM> chunking_result_cond;
	array<bool, STREAM_NUM> chunking_result_obsolete;

	
	/*Obsolete*/
	//chunk hashing
	array<deque<uchar*>, STREAM_NUM> chunk_hash_values;
	array<deque<tuple<char*, uint>>, STREAM_NUM> partitioned_chunks;
	array<mutex, STREAM_NUM> chunk_hashing_mutex;
	array<condition_variable, STREAM_NUM> chunk_hashing_cond;
	array<bool, STREAM_NUM> chunk_hashing_obsolete;


	//chunk matching 
	uint total_duplication_size = 0;
	thread segmentThreads[FINGERPRINTING_THREAD_NUM];
	//Time
	clock_t start, end, start_r, end_r, start_t, end_t, start_ck, end_ck, start_cp, end_cp, start_ch, end_ch, start_cm, end_cm, start_fp, end_fp;
	double time = 0, time_r = 0, time_t = 0, time_ck = 0, time_cp = 0, time_ch, time_cm, time_fp = 0;
	
	int CUDA_Pipeline_Main(int argc, char* argv[]) {
		cout << "\n============ CUDA Implementation With Pipeline and Circular Queue ============\n";
		if (argc != 2) {
			printf("Usage: %s <filename>\n", argv[0]);
			return -1;
		}

		//Initialize file reader
		fin = ifstream(argv[1], ios::in | ios::binary | ios::ate);
		if (!fin.is_open()) {
			cout << "Can not open file " << argv[1] << endl;
			return -1;
		}
		file_len = fin.tellg();
		fin.seekg(0, fin.beg);
		cout << "File size: " << file_len / 1024 << " KB\n";

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
			chunking_result_obsolete[i] = true;
		}
		//initialize chunk hashing
		for (int i = 0; i < STREAM_NUM; ++i) {
			chunk_hashing_obsolete[i] = true;
		}

		start = clock();

		//Create threads
		thread tReadFile(ReadFile);
		thread tTransfer(Transfer);
		thread tChunkingKernel(ChunkingKernel);
		thread tChunkingResultProc(ChunkingResultProc);
		/*thread tChunkHashing(ChunkHashing);
		thread tChunkMatching(ChunkMatching);*/  
		thread tFingerprinting;
		tFingerprinting = thread(MultiFingerprinting);

		tReadFile.join();
		tTransfer.join();
		tChunkingKernel.join();
		tChunkingResultProc.join();
		/*tChunkHashing.join();
		tChunkMatching.join();*/
		tFingerprinting.join();

		end = clock();
		time = (end - start) * 1000 / CLOCKS_PER_SEC;
		
		printf("Read file time: %f ms\n", time_r);
		printf("Transfer time: %f ms\n", time_t);
		printf("Chunking kernel time: %f ms\n", time_ck);
		printf("Chunking processing time: %f ms\n", time_cp);
		printf("Chunk hashing time: %f ms\n", time_ch);
		printf("Chunk matching time %f ms\n", time_cm);
		//printf("Fingerprinting time: %f ms\n", time_fp);
		printf("Total time: %f ms\n", time);
		printf("Found %d bytes of redundancy, which is %f percent of file\n", total_duplication_size, total_duplication_size * 100.0 / file_len);
		
		for (int i = 0; i < STREAM_NUM; ++i) {
			cudaStreamDestroy(stream[i]);
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
			for (uint j = 0; j < result_host_len[resultHostIdx]; ++j) {
				if ((result_host[resultHostIdx][j] & P_MINUS) == 0) {
					chunking_result[streamIdx].push_back(j);
				}
			}

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

	/*Obsolete*/
	void ChunkHashing() {
		int pagableBufferIdx = 0;
		int chunkingResultIdx = 0;
		while (true) {
			//Get pagable buffer ready
			unique_lock<mutex> pagableLock(pagable_buffer_mutex[pagableBufferIdx]);
			while (pagable_buffer_obsolete[pagableBufferIdx] == true) {
				if (chunking_proc_end) {
					chunk_hashing_end = true;
					return;
				}
				pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
			}
			//Get the chunking result ready
			unique_lock<mutex> chunkingResultLock(chunking_result_mutex[chunkingResultIdx]);
			while (chunking_result_obsolete[chunkingResultIdx] == true) {
				if (chunking_proc_end) {
					chunk_hashing_end = true;
					return;
				}
				chunking_result_cond[chunkingResultIdx].wait(chunkingResultLock);
			}
			//Get the chunk hash values and partitioned chunks ready
			unique_lock<mutex> chunkHasingLock(chunk_hashing_mutex[chunkingResultIdx]);
			while (chunk_hashing_obsolete[chunkingResultIdx] == false) {
				chunk_hashing_cond[chunkingResultIdx].wait(chunkHasingLock);
			}

			start_ch = clock();
			re.ChunkHashing(chunking_result[chunkingResultIdx], pagable_buffer[pagableBufferIdx], 
				chunk_hash_values[chunkingResultIdx], partitioned_chunks[chunkingResultIdx]);

			chunking_result[chunkingResultIdx].clear();
			chunking_result_obsolete[chunkingResultIdx] = true;
			chunk_hashing_obsolete[chunkingResultIdx] = false;
			pagableLock.unlock();
			pagable_buffer_obsolete[pagableBufferIdx] = true;
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[chunkingResultIdx].notify_one();
			chunkHasingLock.unlock();
			chunk_hashing_cond[chunkingResultIdx].notify_one();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			chunkingResultIdx = (chunkingResultIdx + 1) % STREAM_NUM;
			end_ch = clock();
			time_ch += (end_ch - start_ch) * 1000 / CLOCKS_PER_SEC;
		}
	}

	/*Obsolete*/
	void ChunkMatching() {
		int pagableBufferIdx = 0;
		int chunkingResultIdx = 0;
		while (true) {
			//No need to lock the pagable buffer
			//Get the chunk hash values and partitioned chunks ready
			unique_lock<mutex> chunkHashingLock(chunk_hashing_mutex[chunkingResultIdx]);
			while (chunk_hashing_obsolete[chunkingResultIdx] == true) {
				if (chunk_hashing_end) {
					return;
				}
				chunk_hashing_cond[chunkingResultIdx].wait(chunkHashingLock);
			}

			start_cm = clock();
			total_duplication_size += re.ChunkMatching(chunk_hash_values[chunkingResultIdx], partitioned_chunks[chunkingResultIdx]);

			//Set pagable buffer to obsolete and notify the thread waiting fot it
			chunk_hashing_obsolete[chunkingResultIdx] = true;
			chunkHashingLock.unlock();
			chunk_hashing_cond[chunkingResultIdx].notify_one();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			chunkingResultIdx = (chunkingResultIdx + 1) % STREAM_NUM;
			end_cm = clock();
			time_cm += (end_cm - start_cm) * 1000 / CLOCKS_PER_SEC;
		}
	}

	void Fingerprinting() {
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

			start_fp = clock();
			total_duplication_size += re.fingerPrinting(chunking_result[chunkingResultIdx], pagable_buffer[pagableBufferIdx]);

			chunking_result[chunkingResultIdx].clear();
			pagable_buffer_obsolete[pagableBufferIdx] = true;
			chunking_result_obsolete[chunkingResultIdx] = true;
			pagableLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[chunkingResultIdx].notify_one();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			chunkingResultIdx = (chunkingResultIdx + 1) % STREAM_NUM;
			end_fp = clock();
			time_fp += (end_fp - start_fp) * 1000 / CLOCKS_PER_SEC;
		}
	}

	void MultiFingerprinting() {
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

			start_fp = clock();
			for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
				segmentThreads[i] = thread(FingerprintingSegment, pagableBufferIdx, chunkingResultIdx, i);
			}

			for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
				segmentThreads[i].join();
			}

			chunking_result[chunkingResultIdx].clear();
			pagable_buffer_obsolete[pagableBufferIdx] = true;
			chunking_result_obsolete[chunkingResultIdx] = true;
			pagableLock.unlock();
			pagable_buffer_cond[pagableBufferIdx].notify_one();
			chunkingResultLock.unlock();
			chunking_result_cond[chunkingResultIdx].notify_one();

			pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
			chunkingResultIdx = (chunkingResultIdx + 1) % STREAM_NUM;
			end_fp = clock();
			time_fp += (end_fp - start_fp) * 1000 / CLOCKS_PER_SEC;
		}
	}

	void FingerprintingSegment(int bufferIdx, int chunkingResultIdx, int segmentNum) {
		uint listSize = chunking_result[chunkingResultIdx].size();
		deque<uint>::iterator begin = chunking_result[chunkingResultIdx].begin();
		deque<uint>::iterator end = chunking_result[chunkingResultIdx].begin();
		advance(begin, segmentNum * listSize / FINGERPRINTING_THREAD_NUM);
		advance(end, min((segmentNum + 1) * listSize / FINGERPRINTING_THREAD_NUM, listSize));
		deque<uint> input(begin, end);
		total_duplication_size += re.fingerPrinting(input, pagable_buffer[bufferIdx]);
	}
}