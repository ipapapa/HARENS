#pragma once
#include "RabinHash.h"
#include "CircularHash.h"
#include "Definition.h"
class RedundancyEliminator_CPP {
private:
	RabinHash hashFunc;
	CircularHash circHash;

	//Add a new chunk into cache, if hash value queue is full also delete the oldest chunk
	void addNewChunk(unsigned char* hashValue, char* chunk, unsigned int chunkSize, bool isDuplicate);
	inline void computeChunkHash(char* chunk, unsigned int chunkSize, unsigned char *hashValue);
		
public:
	deque<unsigned int> chunking(char* package, unsigned int packageSize);
	unsigned int fingerPrinting(deque<unsigned int> indexQ, char* package);
	unsigned int eliminateRedundancy(char* package, unsigned int packageSize);
	RedundancyEliminator_CPP();
	void SetupRedundancyEliminator_CPP();
	~RedundancyEliminator_CPP();
};
