#include "CUDAExecution.h" 
using namespace std;

namespace CUDA_Namespace {

	//constants
	const unsigned int MAX_BUFFER_LEN = MAX_KERNEL_INPUT_LEN;
	const unsigned int MAX_WINDOW_NUM = MAX_BUFFER_LEN - WINDOW_SIZE + 1;
	//file
	ifstream ifs;
	PcapReader fileReader;
	bool readFirstTime = true;
	unsigned int file_length;
	unsigned int cur_file_pos = 0;
	FixedSizedCharArray charArrayBuffer(MAX_BUFFER_LEN);
	char overlap[WINDOW_SIZE - 1];
	//pagable buffer
	char* pagable_buffer;
	unsigned int pagable_buffer_len;
	//fixed buffer
	char* fixed_buffer;
	unsigned int fixed_buffer_len;
	//RedundancyEliminator_CUDA
	RedundancyEliminator_CUDA re;
	//chunking kernel asynchronize
	char* input_kernel;
	unsigned long long* result_kernel;
	unsigned long long* result_host;
	unsigned int result_host_len;
	//chunking result processing
	cudaStream_t stream;
	unsigned int* chunking_result;
	unsigned int chunking_result_len;
	//chunk matching 
	CircularHash hash_pool(MAX_CHUNK_NUM);
	unsigned int total_duplication_size = 0;
	//Time
	clock_t start, end, start_r, end_r, start_t, end_t, start_ck, end_ck, start_cp, end_cp, start_ch, end_ch, start_cm, end_cm;
	double time = 0, time_r = 0, time_t = 0, time_ck = 0, time_cp = 0, time_ch, time_cm;

	int CUDAExecute() {
		IO::Print("\n============================ CUDA Implementation ============================\n");

		re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::NonMultifingerprint);
		//initialize pagable buffer
		pagable_buffer = new char[MAX_BUFFER_LEN];
		//initialize fixed buffer
		cudaMallocHost((void**)&fixed_buffer, MAX_BUFFER_LEN);
		//initialize chunking kernel ascychronize
		cudaMalloc((void**)&input_kernel, MAX_BUFFER_LEN);
		cudaMalloc((void**)&result_kernel, MAX_WINDOW_NUM * BYTES_IN_ULONG);
		cudaMallocHost((void**)&result_host, MAX_WINDOW_NUM * BYTES_IN_ULONG);
		//initialize chunking result processing
		chunking_result = new unsigned int[MAX_WINDOW_NUM];


		bool keepReading = true;
		do {
			keepReading = ReadFile();
			start = clock();
			ChunkingKernel();
			ChunkingResultProc();
			ChunkMatch();
			end = clock();
			time += (end - start) * 1000 / CLOCKS_PER_SEC;
		} while (keepReading);

		IO::Print("Read file time: %f ms\n", time_r);
		//printf("Transfer time: %f ms\n", time_t);
		IO::Print("Chunking kernel time: %f ms\n", time_ck);
		IO::Print("Chunking processing time: %f ms\n", time_cp);
		IO::Print("Chunk hashing time: %f ms\n", time_ch);
		IO::Print("Total time: %f ms\n", time);
		IO::Print("Found %s of redundancy, which is %f percent of file\n"
			, IO::InterpretSize(total_duplication_size)
			, total_duplication_size * 100.0 / file_length);

		//destruct chunking result proc
		delete[] chunking_result;
		//destruct chunking kernel ascychronize
		cudaFree(input_kernel);
		cudaFree(result_kernel);
		cudaFreeHost(result_host);
		//destruct fixed buffer
		cudaFreeHost(fixed_buffer);
		//destruct pagable buffer
		delete[] pagable_buffer;
		return 0;
	}

	bool ReadFile() {
		int curWindowNum;
		//Read the first part
		if (readFirstTime) {
			readFirstTime = false;
			start_r = clock();
			if (IO::FILE_FORMAT == PlainText) {
				ifs = ifstream(IO::input_file_name, ios::in | ios::binary | ios::ate);
				if (!ifs.is_open()) {
					printf("Can not open file %s\n", IO::input_file_name);
					return false;
				}

				file_length = ifs.tellg();
				ifs.seekg(0, ifs.beg);
				IO::Print("File size: %s\n", IO::InterpretSize(file_length));
				pagable_buffer_len = min(MAX_BUFFER_LEN, file_length - cur_file_pos);
				curWindowNum = pagable_buffer_len - WINDOW_SIZE + 1;
				ifs.read(pagable_buffer, pagable_buffer_len);
				cur_file_pos += curWindowNum;

				return pagable_buffer_len == MAX_BUFFER_LEN;
			}
			else if (IO::FILE_FORMAT == Pcap) {
				fileReader.SetupPcapHandle(IO::input_file_name);
				fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN);
				pagable_buffer_len = charArrayBuffer.GetLen();
				memcpy(pagable_buffer, charArrayBuffer.GetArr(), pagable_buffer_len);
				file_length += pagable_buffer_len;

				return pagable_buffer_len == MAX_BUFFER_LEN;
			}
			else
				fprintf(stderr, "Unknown file format\n");

			memcpy(overlap, &pagable_buffer[pagable_buffer_len - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
			time_r += ((float)clock() - start_r) * 1000 / CLOCKS_PER_SEC;
		}
		else { //Read the rest
			if (IO::FILE_FORMAT == PlainText) {
				start_r = clock();
				pagable_buffer_len = min(MAX_BUFFER_LEN, file_length - cur_file_pos + WINDOW_SIZE - 1);
				curWindowNum = pagable_buffer_len - WINDOW_SIZE + 1;
				memcpy(pagable_buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
				ifs.read(&pagable_buffer[WINDOW_SIZE - 1], curWindowNum);
				memcpy(overlap, &pagable_buffer[curWindowNum], WINDOW_SIZE - 1);	//copy the last window into overlap
				cur_file_pos += curWindowNum;
				time_r += ((float)clock() - start_r) * 1000 / CLOCKS_PER_SEC;

				if (pagable_buffer_len != MAX_BUFFER_LEN)
					ifs.close();
				return pagable_buffer_len == MAX_BUFFER_LEN;
			}
			else if (IO::FILE_FORMAT == Pcap) {
				start_r = clock();
				fileReader.ReadPcapFileChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
				memcpy(pagable_buffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
				memcpy(&pagable_buffer[WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
				pagable_buffer_len = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
				file_length += charArrayBuffer.GetLen();
				memcpy(overlap, &pagable_buffer[charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
				time_r += ((float)clock() - start_r) * 1000 / CLOCKS_PER_SEC;

				if (pagable_buffer_len != MAX_BUFFER_LEN)
					IO::Print("File size: %s\n", IO::InterpretSize(file_length));
				return pagable_buffer_len == MAX_BUFFER_LEN;
			}
			else
				fprintf(stderr, "Unknown file format\n");
		}
	}

	void ChunkingKernel() {
		start_ck = clock();
		fixed_buffer_len = pagable_buffer_len;
		memcpy(fixed_buffer, pagable_buffer, fixed_buffer_len);

		re.RabinHashAsync(input_kernel, fixed_buffer, fixed_buffer_len,
			result_kernel, result_host, stream);

		result_host_len = fixed_buffer_len - WINDOW_SIZE + 1;
		end_ck = clock();
		time_ck += (end_ck - start_ck) * 1000 / CLOCKS_PER_SEC;
	}

	void ChunkingResultProc() {
		cudaStreamSynchronize(stream);

		start_cp = clock();
		//all the inputs other than the last one contains #MAX_WINDOW_NUM of windows
		int chunkingResultIdx = 0;
		unsigned int resultHostLen = result_host_len;
		for (unsigned int j = 0; j < resultHostLen; ++j) {
			if ((result_host[j] & P_MINUS) == 0) {
				chunking_result[chunkingResultIdx++] = j;
			}
		}

		chunking_result_len = chunkingResultIdx;

		end_cp = clock();
		time_cp += (end_cp - start_cp) * 1000 / CLOCKS_PER_SEC;
	}

	void ChunkMatch() {
		start_ch = clock();

		total_duplication_size += re.fingerPrinting(chunking_result, chunking_result_len, pagable_buffer);

		end_ch = clock();
		time_ch += (end_ch - start_ch) * 1000 / CLOCKS_PER_SEC;
	}
}