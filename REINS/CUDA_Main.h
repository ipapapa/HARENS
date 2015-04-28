#pragma once
#include "RabinHash.h"
#include "RedundancyEliminator_CUDA.h"

namespace CUDA_Namespace {

	void CUDA_TestOfRabinFingerprint(char* fileContent, uint fileContentLen);

	int CUDA_Main(int argc, char* argv[]);
}