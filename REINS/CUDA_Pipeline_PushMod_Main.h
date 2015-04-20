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

namespace CUDA_Pipeline_PushMod_Namespace {

	void inline Boost();
	void Work(int threadIdx);
	void FingerprintingSegment(int threadIdx, int segmentNum);
	void RoundQuery();

	int CUDA_Pipeline_PushMod_Main(int argc, char* argv[]);
}