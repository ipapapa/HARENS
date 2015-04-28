#pragma once
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"

namespace CPP_Namespace {

	/* old function
	void TestOfRabinHash(char* fileContent, int fileContentLen);
	*/

	void CPP_TestOfRabinFingerprint(char* fileContent, uint fileContentLen);

	int CPP_Main(int argc, char* argv[]);
}