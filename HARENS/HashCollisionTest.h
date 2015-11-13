#pragma once
#include "IO.h"
#include "RabinHash.h"
#include "RedundancyEliminator_CPP_CollisionTest.h"
#include "PcapReader.h"

/*
This class is the module of naive cpp implementation
*/
class HashCollisionTest {
private:
	int hashFuncUsed;
	bool isCollisionCheck;
	unsigned int fileLength = 0;
	RedundancyEliminator_CPP_CollisionTest re;
	ifstream ifs;
	PcapReader fileReader;
	unsigned int curFilePos = 0;

	//shared data
	bool readFirstTime = true;
	char overlap[WINDOW_SIZE - 1];
	char* buffer;
	FixedSizedCharArray charArrayBuffer;

	unsigned int bufferLen = 0;
	deque<unsigned int> chunkingResult;
	//Result
	unsigned int totalDuplicationSize = 0;
	unsigned int totalFalseReportSize = 0;
	//Time
	clock_t start_fin;
	float totFin = 0;

public:
	HashCollisionTest(int hashFuncUsed, bool isCollisionCheck);
	~HashCollisionTest();

	bool ReadFile();
	void Chunking();
	void Fingerprinting();

	int Execute();
};