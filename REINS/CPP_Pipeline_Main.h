#pragma once
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"

namespace CPP_Pipeline_Namespace {

	/* old function
	void TestOfRabinHash(char* fileContent, int fileContentLen);
	*/

	void CPP_ReadFile();
	void CPP_Chunking();
	void CPP_Fingerprinting();

	int CPP_Pipeline_Main(int argc, char* argv[]);
}