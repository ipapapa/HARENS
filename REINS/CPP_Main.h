#pragma once
#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <string>
#include <time.h>
#include "RabinHash.h"
#include "RedundancyEliminator_CPP.h"

namespace CPP_Namespace {

	/* old function
	void TestOfRabinHash(char* fileContent, int fileContentLen);
	*/

	void CPP_TestOfRabinFingerprint(char* fileContent, uint fileContentLen);

	int CPP_Main(int argc, char* argv[]);
}