#pragma once
#include "RabinHash.h"
#include "CircularHash.h"
#include "Definition.h"
class RedundancyEliminator_CPP {
private:
	RabinHash hashFunc;
	CircularHash circHash;

	//Add a new chunk into cache, if hash value queue is full also delete the oldest chunk
	void addNewChunk(ulong hashValue, char* chunk, uint chunkSize, bool isDuplicate);
	inline ulong computeChunkHash(char* chunk, uint chunkSize);
		
public:
	deque<uint> chunking(char* package, uint packageSize);
	uint fingerPrinting(deque<uint> indexQ, char* package);
	uint eliminateRedundancy(char* package, uint packageSize);
	RedundancyEliminator_CPP();
	void SetupRedundancyEliminator_CPP();
	~RedundancyEliminator_CPP();
};
