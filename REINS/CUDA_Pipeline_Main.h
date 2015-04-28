#pragma once
#include <cuda_runtime_api.h> 
#include <cuda.h>
#include "CircularQueue.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

namespace CUDA_Pipeline_Namespace {

	void ReadFile();
	void Transfer();
	void ChunkingKernel();
	void ChunkingResultProc();
	void ChunkHashing();
	void ChunkSegmentHashing(int chunkingResultIdx, int segmentNum);
	void RoundQu1ery();

	int CUDA_Pipeline_Main(int argc, char* argv[]);
}