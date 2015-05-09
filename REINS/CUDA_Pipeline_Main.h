#pragma once
#include <cuda_runtime_api.h> 
#include <cuda.h>
#include <iostream>
#include "CircularPairQueue.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"
#include "CircularTreeHash.h"

namespace CUDA_Pipeline_Namespace {

	void ReadFile();
	void Transfer();
	void ChunkingKernel();
	void ChunkingResultProc();
	void ChunkHashing();
	void ChunkSegmentHashing(int pagableBufferIdx, int chunkingResultIdx, int segmentNum);
	void RoundQuery();

	int CUDA_Pipeline_Main(int argc, char* argv[]);
}