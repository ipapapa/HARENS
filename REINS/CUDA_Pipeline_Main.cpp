#include "CUDA_Pipeline_Main.h"
using namespace std;

namespace CUDA_Pipeline_Namespace {
	
	//constants
	const unsigned int MAX_BUFFER_LEN = MAX_KERNEL_INPUT_LEN;
	const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;
	const int PAGABLE_BUFFER_NUM = 550;
	const int FIXED_BUFFER_NUM = 3;
	const int STREAM_NUM = 3;
	const int FINGERPRINTING_THREAD_NUM = 6;
	const int CIRC_Q_POOL_SIZE = 256;	//Better be power of 2, it would make the module operation faster
	//determin if one thread is end
	bool read_file_end = false; 
	/*bool transfer_end = false;*/
	bool chunking_kernel_end = false;
	bool chunking_proc_end = false;
	bool chunk_hashing_end = false;
	mutex read_file_end_mutex, chunking_kernel_end_mutex, chunking_proc_end_mutex, chunk_hashing_end_mutex;
	//file
	char* fileName;
	ifstream ifs;
	unsigned int file_length;
	FixedSizedCharArray charArrayBuffer(MAX_BUFFER_LEN);
	char overlap[WINDOW_SIZE - 1];
	//pagable buffer
	array<char*, PAGABLE_BUFFER_NUM> pagable_buffer;
	array<unsigned int, PAGABLE_BUFFER_NUM> pagable_buffer_len;
	array<mutex, PAGABLE_BUFFER_NUM> pagable_buffer_mutex;
	array<condition_variable, PAGABLE_BUFFER_NUM> pagable_buffer_cond;
	array<bool, PAGABLE_BUFFER_NUM> pagable_buffer_obsolete;
	//fixed buffer
	array<char*, FIXED_BUFFER_NUM> fixed_buffer;
	array<unsigned int, FIXED_BUFFER_NUM> fixed_buffer_len;
	/*array<mutex, FIXED_BUFFER_NUM> fixed_buffer_mutex;
	array<condition_variable, FIXED_BUFFER_NUM> fixed_buffer_cond;
	array<bool, FIXED_BUFFER_NUM> fixed_buffer_obsolete;*/
	//RedundancyEliminator_CUDA
	RedundancyEliminator_CUDA re;
	//chunking kernel asynchronize
	array<char*, FIXED_BUFFER_NUM> input_kernel;
	array<unsigned long long*, FIXED_BUFFER_NUM> result_kernel;
	array<unsigned long long*, FIXED_BUFFER_NUM> result_host;
	array<unsigned int, FIXED_BUFFER_NUM> result_host_len;
	array<mutex, FIXED_BUFFER_NUM> result_host_mutex;
	array<condition_variable, FIXED_BUFFER_NUM> result_host_cond;
	array<bool, FIXED_BUFFER_NUM> result_host_obsolete;
	array<bool, FIXED_BUFFER_NUM> result_host_executing;
	//chunking result processing
	array<cudaStream_t, STREAM_NUM> stream;
	array<unsigned int*, STREAM_NUM> chunking_result;
	array<unsigned int, STREAM_NUM> chunking_result_len;
	array<mutex, STREAM_NUM> chunking_result_mutex;
	array<condition_variable, STREAM_NUM> chunking_result_cond;
	array<bool, STREAM_NUM> chunking_result_obsolete;
	//chunk hashing
	array<thread, FINGERPRINTING_THREAD_NUM> segment_threads;
	CircularQueuePool<tuple<unsigned long long, unsigned int>> chunk_hash_queue_pool(CIRC_Q_POOL_SIZE);
	//chunk matching 
	array<thread, CIRC_Q_POOL_SIZE> chunk_match_threads;
	vector<CircularHash> circ_hash_pool(CIRC_Q_POOL_SIZE, CircularHash(MAX_CHUNK_NUM / CIRC_Q_POOL_SIZE * 2));
	vector<unsigned int> duplication_size(CIRC_Q_POOL_SIZE, 0);
	unsigned int total_duplication_size = 0;
	//Time
	clock_t start, end, start_r, end_r, start_ck, end_ck, start_cp, end_cp, start_ch, end_ch, start_cm, end_cm;
	double time = 0, time_r = 0, time_ck = 0, time_cp = 0, time_ch, time_cm;
	int count = 0;
	
	int CUDA_Pipeline_Main(int argc, char* argv[]) {
		cout << "\n======= CUDA Implementation With Pipeline and Single Machine MapReduce ======\n";
		if (argc != 2) {
			printf("Usage: %s <filename>\n", argv[0]);
			system("pause");
			return -1;
		}

		fileName = argv[1];

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
		}
		//initialize chunking result processing
		for (int i = 0; i < STREAM_NUM; ++i) {
			cudaStreamCreate(&stream[i]);
			chunking_result[i] = new unsigned int[MAX_WINDOW_NUM];
			chunking_result_obsolete[i] = true;
		}
		//initialize chunk hashing
		//for (int i = 0; i < STREAM_NUM; ++i) {
		//	segment_threads[i] = new thread[FINGERPRINTING_THREAD_NUM];
		//	for (int j = 0; j < FINGERPRINTING_THREAD_NUM; ++j) {
		//		//MAX_WINDOW_NUM / 4 is a guess of the upper bound of the number of chunks
		//		/*chunk_hashing_value_queue[i][j] = CircularUcharArrayQueue(MAX_WINDOW_NUM / 4);
		//		chunk_len_queue[i][j] = CircularUintQueue(MAX_WINDOW_NUM / 4);*/
		//	}
		//}


		//Create threads
		thread tReadFile(ReadFile);
		tReadFile.join();
		start = clock();
		//thread tTransfer(Transfer);
		thread tChunkingKernel(ChunkingKernel);
		thread tChunkingResultProc(ChunkingResultProc);
		thread tChunkHashing(ChunkHashing);
		for (int i = 0; i < CIRC_Q_POOL_SIZE; ++i)
			chunk_match_threads[i] = thread(ChunkMatch, i);

		//tTransfer.join();
		tChunkingKernel.join();
		tChunkingResultProc.join();
		tChunkHashing.join();
		for (int i = 0; i < CIRC_Q_POOL_SIZE; ++i)
			chunk_match_threads[i].join();
		//tRoundQuery.join();

		end = clock();
		time = (end - start) * 1000 / CLOCKS_PER_SEC;
		printf("Read file time: %f ms\n", time_r);
		//printf("Transfer time: %f ms\n", time_t);
		printf("Chunking kernel time: %f ms\n", time_ck);
		printf("Chunking processing time: %f ms\n", time_cp);
		printf("Map (Chunk hashing) time: %f ms\n", time_ch);
		printf("Reduce (Chunk matching) time %f ms\n", time_cm);
		printf("Total time: %f ms\n", time);
		for (int i = 0; i < CIRC_Q_POOL_SIZE; ++i)
			total_duplication_size += duplication_size[i];
		printf("Found %d bytes of redundancy, which is %f percent of file\n", total_duplication_size, total_duplication_size * 100.0 / file_length);

		//destruct chunk hashing & matching
		/*for (int i = 0; i < STREAM_NUM; ++i) {
			delete[] segment_threads[i];
		}*/
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
		unsigned int curFilePos = 0;
		int curWindowNum;
		PcapReader fileReader;
		//Read the first part
		unique_lock<mutex> readFileInitLock(pagable_buffer_mutex[pagableBufferIdx]);
		start_r = clock();
		if (FILE_FORMAT == PlainText) {
			ifs = ifstream(fileName, ios::in | ios::binary | ios::ate);
			if (!ifs.is_open()) {
				cout << "Can not open file " << fileName << endl;
			}
			file_length = ifs.tellg();
			ifs.seekg(0, ifs.beg);
			cout << "File size: " << file_length / 1024 << " KB\n";
			pagable_buffer_len[pagableBufferIdx] = min(MAX_BUFFER_LEN, file_length - curFilePos);
			curWindowNum = pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1;
			ifs.read(pagable_buffer[pagableBufferIdx], pagable_buffer_len[pagableBufferIdx]);
			curFilePos += curWindowNum;
		}
		else if (FILE_FORMAT == Pcap) {
			++count;
			fileReader.SetupPcapHandle(fileName);
			fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN);
			pagable_buffer_len[pagableBufferIdx] = charArrayBuffer.GetLen();
			memcpy(pagable_buffer[pagableBufferIdx], charArrayBuffer.GetArr(), pagable_buffer_len[pagableBufferIdx]);
			file_length += pagable_buffer_len[pagableBufferIdx];
		}
		else
			fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);

		memcpy(overlap, &pagable_buffer[pagableBufferIdx][pagable_buffer_len[pagableBufferIdx] - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
		pagable_buffer_obsolete[pagableBufferIdx] = false;
		readFileInitLock.unlock();
		pagable_buffer_cond[pagableBufferIdx].notify_one();
		pagableBufferIdx = (pagableBufferIdx + 1) % PAGABLE_BUFFER_NUM;
		end_r = clock();
		time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		//Read the rest
		if (FILE_FORMAT == PlainText) {
			//curWindowNum will be less than MAX_BUFFER_LEN - WINDOW_SIZE + 1 when reading the last part
			while (curWindowNum == MAX_WINDOW_NUM) {
				unique_lock<mutex> readFileIterLock(pagable_buffer_mutex[pagableBufferIdx]);
				while (pagable_buffer_obsolete[pagableBufferIdx] == false) {
					pagable_buffer_cond[pagableBufferIdx].wait(readFileIterLock);
				}
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
		else if (FILE_FORMAT == Pcap) {
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
			cout << "File size: " << file_length / 1024 << " KB\n";
			cout << "Need " << count << " pagable buffers\n";
		}
		else
			fprintf(stderr, "Unknown file format %s\n", FILE_FORMAT_TEXT[FILE_FORMAT]);
		unique_lock<mutex> readFileEndLock(read_file_end_mutex);
		read_file_end = true;
		cout << "Reading end\n";
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

	void ChunkingKernel() {
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
					cout << "chunking kernel end\n";
					return;
				}
				readFileEndLock.unlock();
				pagable_buffer_cond[pagableBufferIdx].wait(pagableLock);
			}

			//Get result host ready
			unique_lock<mutex> resultHostLock(result_host_mutex[fixedBufferIdx]);
			while (result_host_obsolete[fixedBufferIdx] == false) {
				cout << "waiting\n";
				result_host_cond[fixedBufferIdx].wait(resultHostLock);
				cout << "done waiting\n";
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
			result_host_executing[resultHostIdx] = false;
			result_host_obsolete[resultHostIdx] = false;
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
			for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
				segment_threads[i] = thread(ChunkSegmentHashing, pagableBufferIdx, chunkingResultIdx, i);
			}

			for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
				segment_threads[i].join();
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

	void ChunkSegmentHashing(int pagableBufferIdx, int chunkingResultIdx, int segmentNum) {
		int listSize = chunking_result_len[chunkingResultIdx];
		unsigned int* chunkingResultSeg = &chunking_result[chunkingResultIdx][segmentNum * listSize / FINGERPRINTING_THREAD_NUM];
		int segLen = listSize / FINGERPRINTING_THREAD_NUM;
		if ((segmentNum + 1) * listSize / FINGERPRINTING_THREAD_NUM > listSize)
			segLen = listSize - segmentNum * listSize / FINGERPRINTING_THREAD_NUM;
		re.ChunkHashingAscynWithCircularQueuePool(chunkingResultSeg, segLen, pagable_buffer[pagableBufferIdx],
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

	void ChunkMatch(int hashPoolIdx) {
		unsigned long long toBeDel;
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

			tuple<unsigned long long, unsigned int> valLenPair = chunk_hash_queue_pool.Pop(hashPoolIdx);
			if (circ_hash_pool[hashPoolIdx].FindAndAdd(get<0>(valLenPair), toBeDel))
				duplication_size[hashPoolIdx] += get<1>(valLenPair);
			//Do something with toBeDel

			end_cm = clock();
			time_cm += (end_cm - start_cm) * 1000 / CLOCKS_PER_SEC;
		}
	}
}