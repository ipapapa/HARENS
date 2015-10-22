#pragma once
#include <cuda_runtime_api.h> 
#include <cuda.h>
#include "IO.h"
#include "RabinHash.h"
#include "PcapReader.h"
#include "RedundancyEliminator_CUDA.h"
#include "OpenAddressCircularHash.h"

namespace CUDA_Namespace {

	bool ReadFile();
	void ChunkingKernel();
	void ChunkingResultProc();
	void ChunkMatch();

	int CUDAExecute();
}