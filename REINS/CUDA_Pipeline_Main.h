#pragma once
#include <cuda_runtime_api.h> 
#include <cuda.h>
#include <iostream>
#include "CircularQueuePool.h"
#include "RabinHash.h"
#include "PcapReader.h"
#include "RedundancyEliminator_CUDA.h"
#include <cassert>

namespace CUDA_Pipeline_Namespace {

	void ReadFile();
	void Transfer();
	void ChunkingKernel();
	void ChunkingResultProc();
	void ChunkHashing();
	void ChunkMatch(int hashPoolIdx);

	void ChunkSegmentHashing(int pagableBufferIdx, int chunkingResultIdx, int segmentNum);

	int CUDA_Pipeline_Main(int argc, char* argv[]);
}