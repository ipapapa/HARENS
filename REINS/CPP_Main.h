#pragma once
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"
#include "PcapReader.h"

namespace CPP_Namespace {

	/* old function
	void TestOfRabinHash(char* fileContent, int fileContentLen);
	*/

	void CPP_TestOfRabinFingerprint(char* fileContent, unsigned int fileContentLen);

	int CPP_Main(int argc, char* argv[]);
}