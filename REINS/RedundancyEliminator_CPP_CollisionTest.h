#pragma once
#include "RedundancyEliminator_CPP.h"
#include "LRUHash_CollisionTest.h"

class RedundancyEliminator_CPP_CollisionTest: public RedundancyEliminator_CPP
{
private:
	LRUHash_CollisionTest<unsigned long long> circHash;

	unordered_map<unsigned long long, tuple<char*, int>> rabinMap;
	unordered_map<unsigned char*, tuple<char*, int>, CharArrayHashFunc, CharArrayEqualTo> sha1Map;

	void addNewChunk(unsigned long long hashValue, char* chunk
		, unsigned int chunkSize, bool isDuplicate);
public:
	RedundancyEliminator_CPP_CollisionTest();
	void SetupRedundancyEliminator_CPP_CollisionTest();
	~RedundancyEliminator_CPP_CollisionTest();

	inline unsigned long long computeChunkHash(char* chunk, unsigned int chunkSize);
	unsigned int SHA1FingerPrinting(deque<unsigned int> indexQ, char* package);
	unsigned int RabinFingerPrinting(deque<unsigned int> indexQ, char* package);
	tuple<unsigned int, unsigned int> SHA1FingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package);
	tuple<unsigned int, unsigned int> RabinFingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package);
};

