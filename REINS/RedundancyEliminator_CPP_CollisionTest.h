#pragma once
#include "RedundancyEliminator_CPP.h"
#include "LRUHash_CollisionTest.h"

class RedundancyEliminator_CPP_CollisionTest: RedundancyEliminator_CPP
{
private:
	LRUHash_CollisionTest<unsigned long long> circHash;

	void addNewChunk(unsigned long long hashValue, char* chunk
		, unsigned int chunkSize, bool isDuplicate);
public:
	RedundancyEliminator_CPP_CollisionTest();
	~RedundancyEliminator_CPP_CollisionTest();

	inline unsigned long long computeChunkHash(char* chunk, unsigned int chunkSize);
	unsigned int SHA1FingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package);
	unsigned int RabinFingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package);
};

