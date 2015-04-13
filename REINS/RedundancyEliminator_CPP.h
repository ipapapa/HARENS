#pragma once
#include <unordered_set>
#include <list>
#include <deque>
#include <string>
#include <stdio.h>
#include <fstream>
#include <openssl/sha.h>
#include "RabinHash.h"
#include "CircularHash.h"
#include "Definition.h"
class RedundancyEliminator_CPP {
private:
	RabinHash hashFunc;
	CircularHash circHash;

	//Add a new chunk into cache, if hash value queue is full also delete the oldest chunk
	void addNewChunk(uchar* hashValue, char* chunk, uint chunkSize);
	inline uchar* computeChunkHash(char* chunk, uint chunkSize);
		
public:
	deque<uint> chunking(char* package, uint packageSize);
	uint fingerPrinting(deque<uint> indexQ, char* package);
	uint eliminateRedundancy(char* package, uint packageSize);
	RedundancyEliminator_CPP();
	~RedundancyEliminator_CPP();
};
