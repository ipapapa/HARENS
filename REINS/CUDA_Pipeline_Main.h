#pragma once
#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <string>
#include <time.h>
#include <algorithm>
#include <mutex>
#include <thread>
#include <chrono>
#include <deque>
#include <condition_variable>
#include <cuda_runtime_api.h> 
#include <cuda.h>
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

namespace CUDA_Pipeline_Namespace {

	void ReadFile();
	void Transfer();
	void ChunkingKernel();
	void ChunkingResultProc();
	void ChunkHashing();
	void ChunkMatching();
	void Fingerprinting();
	void MultiFingerprinting();
	void FingerprintingSegment(int bufferIdx, int chunkingResultIdx, int segmentNum);

	int CUDA_Pipeline_Main(int argc, char* argv[]);
}