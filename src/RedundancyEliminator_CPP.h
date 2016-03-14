#pragma once
#include "RabinHash.h"
#include "LRUStrHash.h"
#include "Definition.h"
#include "EncryptionHashes.h"

/*
* The redundancy elimination module for non-cuda accelerated implementations
*/
class RedundancyEliminator_CPP {
protected:
	RabinHash hashFunc;
	LRUStrHash<SHA1_HASH_LENGTH> circHash;

	//Add a new chunk into cache, if hash value queue is full also delete the oldest chunk
	void addNewChunk(unsigned char* hashValue,
					 char* chunk, 
					 unsigned int chunkSize, 
					 bool isDuplicate);
		
public:
	RedundancyEliminator_CPP();
	void SetupRedundancyEliminator_CPP();
	~RedundancyEliminator_CPP();

	/*
	* Partition a stream into chunks
	*/
	deque<unsigned int> chunking(char* package, unsigned int packageSize);

	/*
	* Compute hash value for each chunk and find out the duplicate chunks
	*/
	unsigned int fingerPrinting(deque<unsigned int> indexQ, char* package);

	/*
	* Read in a stream, partition it into chunks, and find out the duplicate chunks
	*/
	unsigned int eliminateRedundancy(char* package, unsigned int packageSize);
};
