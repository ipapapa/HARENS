#include "CUDA_Pipeline_PushMod_Main.h"
using namespace std;

namespace CUDA_Pipeline_PushMod_Namespace {
	//constants
	const uint MAX_BUFFER_LEN = MAX_KERNEL_INPUT_LEN;
	const uint MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;
	const int FINGERPRINTING_THREAD_NUM = 4;
	//threads
	int thread_num;
	thread* worker_threads;
	//file
	ifstream fin;
	uint file_len;
	char overlap[WINDOW_SIZE - 1];
	//pagable buffer
	char** pagable_buffer;
	uint* buffer_len;
	//fixed buffer
	char** fixed_buffer;
	//RedundancyEliminator_CUDA
	RedundancyEliminator_CUDA re;
	//rabin hash kernel asynchronize
	char** input_kernel;
	ulong** result_kernel;
	ulong** result_host;
	//chunking
	cudaStream_t* stream;
	uint** chunking_result;
	uint* chunking_result_len;
	//chunk hashing & matching
	uint total_duplication_size = 0;
	thread** segment_threads;
	uchar*** chunk_hashing_value_list;
	uint*** chunk_len_list;	//This is only for simulation, in real case don't need the "uint chunk length"
	mutex** chunk_hash_mutex;
	//Circular querying hash
	CircularHash circ_hash;
	thread ascyn_matching_threads;
	bool kill_ascyn_matching = false;
	mutex kill_ascyn_matching_mutex;
	//Time
	clock_t start, end, start_r, end_r, start_t, end_t, start_rhk, end_rhk, start_c, end_c, start_fp, end_fp;
	double time = 0, time_r = 0, time_t = 0, time_rhk = 0, time_c = 0, time_fp = 0;

	int CUDA_Pipeline_PushMod_Main(int argc, char* argv[]) {
		cout << "\n======== Push Mode CUDA Implementation With Pipeline and Round Query ========\n";
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

		thread_num = (file_len - WINDOW_SIZE + 1 + MAX_WINDOW_NUM - 1) / MAX_WINDOW_NUM;
		worker_threads = new thread[thread_num];

		re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::MultiFingerprint);
		circ_hash.SetupCircularHash(MAX_CHUNK_NUM);
		//initialize pagable buffer
		pagable_buffer = new char*[thread_num];
		buffer_len = new uint[thread_num];
		for (int i = 0; i < thread_num; ++i) {
			pagable_buffer[i] = new char[MAX_BUFFER_LEN];
		}
		//initialize fixed buffer
		fixed_buffer = new char*[thread_num];
		for (int i = 0; i < thread_num; ++i) {
			cudaMallocHost((void**)&fixed_buffer[i], MAX_BUFFER_LEN);
		}
		//initialize chunking kernel ascychronize
		input_kernel = new char*[thread_num];
		result_kernel = new ulong*[thread_num];
		result_host = new ulong*[thread_num];
		for (int i = 0; i < thread_num; ++i) {
			cudaMalloc((void**)&input_kernel[i], MAX_BUFFER_LEN);
			cudaMalloc((void**)&result_kernel[i], MAX_WINDOW_NUM * BYTES_IN_ULONG);
			cudaMallocHost((void**)&result_host[i], MAX_WINDOW_NUM * BYTES_IN_ULONG);
		}
		//initialize chunking
		stream = new cudaStream_t[thread_num];
		chunking_result = new uint*[thread_num];
		chunking_result_len = new uint[thread_num];
		for (int i = 0; i < thread_num; ++i) {
			cudaStreamCreate(&stream[i]);
			chunking_result[i] = new uint[MAX_WINDOW_NUM];
		}
		//initialize chunk hashing & matching
		segment_threads = new thread*[thread_num];
		chunk_hashing_value_list = new uchar**[thread_num];
		chunk_len_list = new uint**[thread_num];
		chunk_hash_mutex = new mutex*[thread_num];
		for (int i = 0; i < thread_num; ++i) {
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

		//Boost the engine
		Boost();
		ascyn_matching_threads = thread(RoundQuery);
		for (int i = 0; i < thread_num; ++i)
			worker_threads[i].join();

		kill_ascyn_matching_mutex.lock();
		kill_ascyn_matching = true;
		kill_ascyn_matching_mutex.unlock();
		ascyn_matching_threads.join();

		end = clock();
		time = (end - start) * 1000 / CLOCKS_PER_SEC;

		printf("Read file time: %f ms\n", time_r);
		printf("Transfer time: %f ms\n", time_t);
		printf("Chunking kernel time: %f ms\n", time_rhk);
		printf("Chunking processing time: %f ms\n", time_c);
		printf("Fingerprinting time: %f ms\n", time_fp);
		printf("Total time: %f ms\n", time);
		printf("Found %d bytes of redundancy, which is %f percent of file\n", total_duplication_size, total_duplication_size * 100.0 / file_len);

		//destruct chunk hashing & matching
		for (int i = 0; i < thread_num; ++i) {
			for (int j = 0; j < FINGERPRINTING_THREAD_NUM; ++j) {
				delete[] chunk_hashing_value_list[i][j];
				delete[] chunk_len_list[i][j];
			}
			delete[] segment_threads[i];
			delete[] chunk_hashing_value_list[i];
			delete[] chunk_len_list[i];
			delete[] chunk_hash_mutex[i];
		}
		delete[] segment_threads;
		delete[] chunk_hashing_value_list;
		delete[] chunk_len_list;
		delete[] chunk_hash_mutex;
		//destruct chunking
		for (int i = 0; i < thread_num; ++i) {
			cudaStreamDestroy(stream[i]);
			delete[] chunking_result[i];
		}
		delete[] stream;
		delete[] chunking_result;
		//destruct rabin hash ascychronize
		for (int i = 0; i < thread_num; ++i) {
			cudaFree(input_kernel[i]);
			cudaFree(result_kernel[i]);
			cudaFreeHost(result_host[i]);
		}
		delete[] result_host;
		delete[] result_kernel;
		delete[] input_kernel;
		//destruct fixed buffer
		for (int i = 0; i < thread_num; ++i) {
			cudaFreeHost(fixed_buffer[i]);
		}
		delete[] fixed_buffer;
		//destruct pagable buffer
		for (int i = 0; i < thread_num; ++i) {
			delete[] pagable_buffer[i];
		}
		delete[] buffer_len;
		delete[] pagable_buffer;
		delete[] worker_threads;
		return 0;
	}

	void inline Boost() {
		int threadIdx = 0;
		uint curFilePos = 0;
		int curWindowNum;

		//Read the first part
		start_r = clock();
		buffer_len[threadIdx] = min(MAX_BUFFER_LEN, file_len - curFilePos);
		curWindowNum = buffer_len[threadIdx] - WINDOW_SIZE + 1;
		fin.read(pagable_buffer[threadIdx], buffer_len[threadIdx]);
		memcpy(overlap, &pagable_buffer[threadIdx][curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
		worker_threads[threadIdx] = thread(Work, threadIdx);
		++threadIdx;
		curFilePos += curWindowNum;
		//Read the rest
		//curWindowNum will be less than MAX_BUFFER_LEN - WINDOW_SIZE + 1 when reading the last part
		while (curWindowNum == MAX_WINDOW_NUM) {
			start_r = clock();
			buffer_len[threadIdx] = min(MAX_BUFFER_LEN, file_len - curFilePos);
			curWindowNum = buffer_len[threadIdx] - WINDOW_SIZE + 1;
			memcpy(pagable_buffer[threadIdx], overlap, WINDOW_SIZE - 1);		//copy the overlap into current part
			fin.read(&pagable_buffer[threadIdx][WINDOW_SIZE - 1], curWindowNum);
			memcpy(overlap, &pagable_buffer[threadIdx][curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
			worker_threads[threadIdx] = thread(Work, threadIdx);
			++threadIdx;
			curFilePos += curWindowNum;
		}
		end_r = clock();
		time_r += (end_r - start_r) * 1000 / CLOCKS_PER_SEC;
		fin.close();
	}

	void Work(int threadIdx) {
		start_t = clock();
		memcpy(fixed_buffer[threadIdx], pagable_buffer[threadIdx], buffer_len[threadIdx]);
		end_t = clock();
		time_t += (end_t - start_t) * 1000 / CLOCKS_PER_SEC;
		start_rhk = clock();
		re.RabinHashAsync(input_kernel[threadIdx], fixed_buffer[threadIdx], buffer_len[threadIdx],
			result_kernel[threadIdx], result_host[threadIdx], stream[threadIdx]);
		cudaStreamSynchronize(stream[threadIdx]);
		end_rhk = clock();
		time_rhk += (end_rhk - start_rhk) * 1000 / CLOCKS_PER_SEC;
		start_c = clock();
		int chunkingResultIdx = 0;
		for (uint j = 0; j < buffer_len[threadIdx] - WINDOW_SIZE + 1; ++j) {
			if ((result_host[threadIdx][j] & P_MINUS) == 0) {
				chunking_result[threadIdx][chunkingResultIdx++] = j;
			}
		}
		chunking_result_len[threadIdx] = chunkingResultIdx;
		end_c = clock();
		time_c += (end_c - start_c) * 1000 / CLOCKS_PER_SEC;
		start_fp = clock();
		for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
			segment_threads[threadIdx][i] = thread(FingerprintingSegment, threadIdx, i);
		}

		for (int i = 0; i < FINGERPRINTING_THREAD_NUM; ++i) {
			segment_threads[threadIdx][i].join();
		}
		end_fp = clock();
		time_fp += (end_fp - start_fp) * 1000 / CLOCKS_PER_SEC;
	}

	void FingerprintingSegment(int threadIdx, int segmentNum) {
		int size = chunking_result_len[threadIdx];
		uint* chunkingResultSeg = &chunking_result[threadIdx][segmentNum * size / FINGERPRINTING_THREAD_NUM];
		int segLen = size / FINGERPRINTING_THREAD_NUM;
		if ((segmentNum + 1) * size / FINGERPRINTING_THREAD_NUM > size)
			segLen = size - segmentNum * size / FINGERPRINTING_THREAD_NUM;
		re.ChunkHashingAscyn(chunkingResultSeg, segLen, pagable_buffer[threadIdx],
			chunk_hashing_value_list[threadIdx][segmentNum], 
			chunk_len_list[threadIdx][segmentNum], chunk_hash_mutex[threadIdx][segmentNum]);
	}

	void RoundQuery() {
		bool noHashValueFound;
		tuple<uchar*, uint> empty = tuple<uchar*, uint>(new uchar(' '), -1);
		uchar* hashValue;
		uint chunkLen;
		while (true) {
			noHashValueFound = true;
			for (int threadIdx = 0; threadIdx < thread_num; ++threadIdx) {
				for (int segmentNum = 0; segmentNum < FINGERPRINTING_THREAD_NUM; ++segmentNum) {
					tuple<uchar*, uint> hashValueAndLen = empty;
					chunk_hash_mutex[threadIdx][segmentNum].lock();
					/*if (!hash_value_queue[threadIdx][segmentNum].empty()) {
						hashValueAndLen = hash_value_queue[threadIdx][segmentNum].front();
						hash_value_queue[threadIdx][segmentNum].pop_front();
						noHashValueFound = false;
					}*/
					chunk_hash_mutex[threadIdx][segmentNum].unlock();

					hashValue = get<0>(hashValueAndLen);
					chunkLen = get<1>(hashValueAndLen);
					if (chunkLen != -1) {
						if (circ_hash.Find(hashValue)) {
							total_duplication_size += chunkLen;
						}
						else {
							uchar* to_be_del = circ_hash.Add(hashValue);
							if (to_be_del != NULL)
								delete[] to_be_del;
							//In real software we are supposed to deal with the chunk in disk
						}
					}
				}
			}
			if (noHashValueFound) {
				kill_ascyn_matching_mutex.lock();
				if (kill_ascyn_matching) {
					kill_ascyn_matching_mutex.unlock();
					return;
				}
				kill_ascyn_matching_mutex.unlock();
				this_thread::sleep_for(chrono::microseconds(500));
			}
		}
	}
}