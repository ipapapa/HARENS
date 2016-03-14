#pragma once
#include "IO.h"
#include "PcapReader.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

class CudaAcceleratedAlg {
private:
	//file
	bool readFirstTime = true;
	unsigned long long totalFileLen;
	FixedSizedCharArray charArrayBuffer;
	char overlap[WINDOW_SIZE - 1];
	//pagable buffer
	char* pagableBuffer;
	unsigned int pagableBufferLen;
	//fixed buffer
	char* fixedBuffer;
	unsigned int fixedBufferLen;
	//RedundancyEliminator_CUDA
	RedundancyEliminator_CUDA re;
	//chunking kernel asynchronize
	char* kernelInputBuffer;
	unsigned int* kernelResultBuffer;
	unsigned int* hostResultBuffer;
	unsigned int hostResultLen;
	//chunking result processing
	cudaStream_t stream;
	unsigned int* chunkingResultBuffer;
	unsigned int chunkingResultLen;
	//chunk matching 
	LRUStrHash<SHA1_HASH_LENGTH> hashPool;
	unsigned long long totalDuplicationSize = 0;
	//Time
	clock_t start, 
			end, 
			startReading, 
			endReading, 
			startChunkingKernel, 
			endChunkingKernel, 
			startChunkPartitioning, 
			endChunkPartitioning, 
			startChunkHashing, 
			endChunkHashing, 
			startChunkMatching, 
			endChunkMatching;
	double timeTotal = 0, 
		   timeReading = 0, 
		   timeChunkingKernel = 0, 
		   timeChunkPartitioning = 0, 
		   timeChunkHashing, 
		   timeChunkMatching;

	bool ReadFile();
	void ChunkingKernel();
	void ChunkingResultProc();
	void ChunkMatch();

public:
	CudaAcceleratedAlg();
	~CudaAcceleratedAlg();

	int Execute();
	void Test(double &rate, double &time);
};