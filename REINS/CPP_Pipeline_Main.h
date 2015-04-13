#pragma once
#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <algorithm>
#include <string>
#include <time.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <stdio.h>
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