#include "CudaAcceleratedAlg.h" 
using namespace std;

CudaAcceleratedAlg::CudaAcceleratedAlg()
	: charArrayBuffer(MAX_BUFFER_LEN), hashPool(MAX_CHUNK_NUM) {
	re.SetupRedundancyEliminator_CUDA(RedundancyEliminator_CUDA::NonMultifingerprint);
	//initialize pagable buffer
	pagableBuffer = new char[MAX_BUFFER_LEN];
	//initialize fixed buffer
	cudaMallocHost((void**)&fixedBuffer, MAX_BUFFER_LEN);
	//initialize chunking kernel ascychronize
	cudaMalloc((void**)&kernelInputBuffer, MAX_BUFFER_LEN);
	cudaMalloc((void**)&kernelResultBuffer, MAX_WINDOW_NUM * BYTES_IN_UINT);
	cudaMallocHost((void**)&hostResultBuffer, MAX_WINDOW_NUM * BYTES_IN_UINT);
	//initialize chunking result processing
	chunkingResultBuffer = new unsigned int[MAX_WINDOW_NUM];
}

CudaAcceleratedAlg::~CudaAcceleratedAlg() {
	//destruct chunking result proc
	delete[] chunkingResultBuffer;
	//destruct chunking kernel ascychronize
	cudaFree(kernelInputBuffer);
	cudaFree(kernelResultBuffer);
	cudaFreeHost(hostResultBuffer);
	//destruct fixed buffer
	cudaFreeHost(fixedBuffer);
	//destruct pagable buffer
	delete[] pagableBuffer;
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
		timeTotal += (end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	IO::Print("Read file time: %f ms\n", timeReading);
	//printf("Transfer time: %f ms\n", time_t);
	IO::Print("Chunking kernel time: %f ms\n", timeChunkingKernel);
	IO::Print("Chunking processing time: %f ms\n", timeChunkPartitioning);
	IO::Print("Chunk hashing time: %f ms\n", timeChunkHashing);
	IO::Print("Total time: %f ms\n", timeTotal);
	IO::Print("Found %s of redundency, "
		, IO::InterpretSize(totalDuplicationSize));
	IO::Print("which is %f %% of file\n"
		, (float)totalDuplicationSize / totalFileLen * 100);

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
		timeTotal += (end - start) * 1000 / CLOCKS_PER_SEC;
	} while (keepReading);

	rate = totalDuplicationSize * 100.0 / totalFileLen;
	time = timeTotal;
}

bool CudaAcceleratedAlg::ReadFile() {
	int curWindowNum;
	//Read the first part
	if (readFirstTime) {
		readFirstTime = false;
		startReading = clock();

		IO::fileReader->SetupReader(IO::input_file_name);
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN);
		pagableBufferLen = charArrayBuffer.GetLen();
		memcpy(pagableBuffer, charArrayBuffer.GetArr(), pagableBufferLen);
		totalFileLen += pagableBufferLen;

		return pagableBufferLen == MAX_BUFFER_LEN;

		memcpy(overlap, &pagableBuffer[pagableBufferLen - WINDOW_SIZE + 1], WINDOW_SIZE - 1);	//copy the last window into overlap
		timeReading += ((float)clock() - startReading) * 1000 / CLOCKS_PER_SEC;
	}
	else { //Read the rest
		startReading = clock();
		IO::fileReader->ReadChunk(charArrayBuffer, MAX_BUFFER_LEN - WINDOW_SIZE + 1);
		memcpy(pagableBuffer, overlap, WINDOW_SIZE - 1);	//copy the overlap into current part
		memcpy(&pagableBuffer[WINDOW_SIZE - 1], charArrayBuffer.GetArr(), charArrayBuffer.GetLen());
		pagableBufferLen = charArrayBuffer.GetLen() + WINDOW_SIZE - 1;
		totalFileLen += charArrayBuffer.GetLen();
		memcpy(overlap, &pagableBuffer[charArrayBuffer.GetLen()], WINDOW_SIZE - 1);	//copy the last window into overlap
		timeReading += ((float)clock() - startReading) * 1000 / CLOCKS_PER_SEC;

		if (pagableBufferLen != MAX_BUFFER_LEN)
			IO::Print("File size: %s\n", IO::InterpretSize(totalFileLen));
		return pagableBufferLen == MAX_BUFFER_LEN;
	}
}

void CudaAcceleratedAlg::ChunkingKernel() {
	startChunkingKernel = clock();
	fixedBufferLen = pagableBufferLen;
	memcpy(fixedBuffer, pagableBuffer, fixedBufferLen);

	re.RabinHashAsync(kernelInputBuffer, 
					  fixedBuffer, 
					  fixedBufferLen,
					  kernelResultBuffer, 
					  hostResultBuffer,
					  stream);

	hostResultLen = fixedBufferLen - WINDOW_SIZE + 1;
	endChunkingKernel = clock();
	timeChunkingKernel += (endChunkingKernel - startChunkingKernel) * 1000 / CLOCKS_PER_SEC;
}

void CudaAcceleratedAlg::ChunkingResultProc() {
	cudaStreamSynchronize(stream);

	startChunkPartitioning = clock();
	//all the inputs other than the last one contains #MAX_WINDOW_NUM of windows
	int chunkingResultIdx = 0;
	unsigned int resultHostLen = hostResultLen;
	for (unsigned int j = 0; j < resultHostLen; ++j) {
		if (hostResultBuffer[j] == 0) {
			chunkingResultBuffer[chunkingResultIdx++] = j;
		}
	}

	chunkingResultLen = chunkingResultIdx;

	endChunkPartitioning = clock();
	timeChunkPartitioning += (endChunkPartitioning - startChunkPartitioning) * 1000 / CLOCKS_PER_SEC;
}

void CudaAcceleratedAlg::ChunkMatch() {
	startChunkHashing = clock();

	totalDuplicationSize += re.fingerPrinting(chunkingResultBuffer, chunkingResultLen, pagableBuffer);

	endChunkHashing = clock();
	timeChunkHashing += (endChunkHashing - startChunkHashing) * 1000 / CLOCKS_PER_SEC;
}