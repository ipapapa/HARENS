#include "CudaAcceleratedAlg.h" 
using namespace std;

CudaAcceleratedAlg::CudaAcceleratedAlg()
	: charArrayBuffer(MAX_BUFFER_LEN), hash_pool(MAX_CHUNK_NUM) {
	re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::NonMultifingerprint);
	//initialize pagable buffer
	pagable_buffer = new char[MAX_BUFFER_LEN];
	//initialize fixed buffer
	cudaMallocHost((void**)&fixed_buffer, MAX_BUFFER_LEN);
	//initialize chunking kernel ascychronize
	cudaMalloc((void**)&input_kernel, MAX_BUFFER_LEN);
	cudaMalloc((void**)&result_kernel, MAX_WINDOW_NUM * BYTES_IN_UINT);
	cudaMallocHost((void**)&result_host, MAX_WINDOW_NUM * BYTES_IN_UINT);
	//initialize chunking result processing
	chunking_result = new unsigned int[MAX_WINDOW_NUM];
}

CudaAcceleratedAlg::~CudaAcceleratedAlg() {
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
}

int CudaAcceleratedAlg::Execute() {
	IO::Print("\n============================ CUDA Implementation ============================\n");

	bool keepReading = true;
	do {
		keepReading = ReadFile();
		start = clock();
		ChunkingKernel();
		ChunkingResultProc();
		ChunkMatch();
		end = clock();
		time_tot += (end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	IO::Print("Read file time: %f ms\n", time_r);
	//printf("Transfer time: %f ms\n", time_t);
	IO::Print("Chunking kernel time: %f ms\n", time_ck);
	IO::Print("Chunking processing time: %f ms\n", time_cp);
	IO::Print("Chunk hashing time: %f ms\n", time_ch);
	IO::Print("Total time: %f ms\n", time_tot);
	IO::Print("Found %s of redundency, "
		, IO::InterpretSize(total_duplication_size));
	IO::Print("which is %f %% of file\n"
		, (float)total_duplication_size / file_length * 100);

	return 0;
}

void CudaAcceleratedAlg::Test(double &rate, double &time) {
	bool keepReading = true;
	do {
		keepReading = ReadFile();
		start = clock();
		ChunkingKernel();
		ChunkingResultProc();
		ChunkMatch();
		end = clock();
		time_tot += (end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	rate = total_duplication_size * 100.0 / file_length;
	time = time_tot;
}

bool CudaAcceleratedAlg::ReadFile() {
	int curWindowNum;
	//Read the first part
	if (readFirstTime) {
		readFirstTime = false;
		start_r = clock();

		IO::fileReader->SetupReader(IO::input_file_name);
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);
		pagable_buffer_len = charArrayBuffer.GetLen();
		memcpy(pagable_buffer, charArrayBuffer.GetArr(), pagable_buffer_len);
		file_length += pagable_buffer_len;

		return pagable_buffer_len == MAX_BUFFER_LEN;

		memcpy(overlap, &pagable_buffer[pagable_buffer_len - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
		time_r += ((float)clock() - start_r) * 1000 / CLOCKS_PER_SEC;
	}
	else { //Read the rest
		start_r = clock();
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
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
}

void CudaAcceleratedAlg::ChunkingKernel() {
	start_ck = clock();
	fixed_buffer_len = pagable_buffer_len;
	memcpy(fixed_buffer, pagable_buffer, fixed_buffer_len);

	re.RabinHashAsync(input_kernel, 
					  fixed_buffer, 
					  fixed_buffer_len,
					  result_kernel, 
					  result_host,
					  stream);

	result_host_len = fixed_buffer_len - WINDOW_SIZE + 1;
	end_ck = clock();
	time_ck += (end_ck - start_ck) * 1000 / CLOCKS_PER_SEC;
}

void CudaAcceleratedAlg::ChunkingResultProc() {
	cudaStreamSynchronize(stream);

	start_cp = clock();
	//all the inputs other than the last one contains #MAX_WINDOW_NUM of windows
	int chunkingResultIdx = 0;
	unsigned int resultHostLen = result_host_len;
	for (unsigned int j = 0; j < resultHostLen; ++j) {
		if (result_host[j] == 0) {
			chunking_result[chunkingResultIdx++] = j;
		}
	}

	chunking_result_len = chunkingResultIdx;

	end_cp = clock();
	time_cp += (end_cp - start_cp) * 1000 / CLOCKS_PER_SEC;
}

void CudaAcceleratedAlg::ChunkMatch() {
	start_ch = clock();

	total_duplication_size += re.fingerPrinting(chunking_result, chunking_result_len, pagable_buffer);

	end_ch = clock();
	time_ch += (end_ch - start_ch) * 1000 / CLOCKS_PER_SEC;
}