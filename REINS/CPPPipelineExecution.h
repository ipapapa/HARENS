#pragma once
#include "IO.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"
#include "PcapReader.h"

namespace CPP_Pipeline_Namespace {

	/* old function
	void TestOfRabinHash(char* fileContent, int fileContentLen);
	*/

	void ReadFile();
	void Chunking();
	void Fingerprinting();

	int CPPPipelineExecute();
}