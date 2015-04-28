#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

namespace CUDA_Pipeline_PushMod_Namespace {

	void inline Boost();
	void Work(int threadIdx);
	void FingerprintingSegment(int threadIdx, int segmentNum);
	void RoundQuery();

	int CUDA_Pipeline_PushMod_Main(int argc, char* argv[]);
}