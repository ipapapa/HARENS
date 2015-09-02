#include "RedundancyEliminator_CPP.h"

RedundancyEliminator_CPP::RedundancyEliminator_CPP() {
	//moved the content to setup function to avoid the duplicated dynamic initializer
}

void RedundancyEliminator_CPP::SetupRedundancyEliminator_CPP() {
	hashFunc = RabinHash();
	circHash.SetupCircularHash(MAX_CHUNK_NUM);
	//The real software need to generate a initial file named 0xFF here
	//Check Circular.cpp to see the reason
}

RedundancyEliminator_CPP::~RedundancyEliminator_CPP() {
	//The real software would delete all the generated files here
}

/*
Add a new chunck into the file system, if the hash value queue is full, also delete the oldest chunk.
*/
void RedundancyEliminator_CPP::addNewChunk(unsigned long long hashValue, char* chunk, unsigned int chunkSize, bool isDuplicate) {
	unsigned long long to_be_del = circHash.Add(hashValue, isDuplicate);
	//fstream file(hashValue.c_str(), std::fstream::in|std::fstream::out);
	////we are actually supposed to do something with chunkSize here
	//file << chunk;
	//file.close();
}

deque<unsigned int> RedundancyEliminator_CPP::chunking(char* package, unsigned int packageSize) {
	deque<unsigned int> indexQ = deque<unsigned int>();
	char* chunk = new char[WINDOW_SIZE];

	for (unsigned int i = 0; i < packageSize - WINDOW_SIZE + 1; ++i) {
		memcpy(chunk, &(package[i]), WINDOW_SIZE);
		unsigned long long windowFingerPrint = hashFunc.Hash(chunk, WINDOW_SIZE);
		if ((windowFingerPrint & P_MINUS) == 0) { // marker found
			indexQ.push_back(i);
		}
	}
	return indexQ;
}

unsigned int RedundancyEliminator_CPP::fingerPrinting(deque<unsigned int> indexQ, char* package) {
	unsigned int duplicationSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	bool isDuplicate;
	for (deque<unsigned int>::const_iterator iter = indexQ.begin(); iter != indexQ.end(); ++iter) {
		if (prevIdx == 0) {
			prevIdx = *iter;
			continue;
		}
		unsigned int chunkLen = *iter - prevIdx;
		//if chunk is too small, combine it with the next chunk
		if (chunkLen < MIN_CHUNK_LEN)	
			continue;

		chunk = &(package[prevIdx]);
		unsigned long long chunkHash = computeChunkHash(chunk, chunkLen);
		if (circHash.Find(chunkHash)) { //find duplications
			duplicationSize += chunkLen;
			isDuplicate = true;
		}
		else {
			isDuplicate = false;
		}
		addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);
		/*unsigned long long toBeDel;
		if (circHash.FindAndAdd(chunkHash, toBeDel))
			duplicationSize += chunkLen;*/

		prevIdx = *iter;
	}
	return duplicationSize;
}

unsigned int RedundancyEliminator_CPP::eliminateRedundancy(char* package, unsigned int packageSize) {
	clock_t start = clock();
	deque<unsigned int> indexQ = chunking(package, packageSize);
	double time = (clock() - start) * 1000 / CLOCKS_PER_SEC;
	printf("Chunk partition time: %f ms\n", time);
	return fingerPrinting(indexQ, package);
}

/*
Compute the hash value of chunk, should use sha3 to avoid collision,
I'm using rabin hash here for convience
*/
inline unsigned long long RedundancyEliminator_CPP::computeChunkHash(char* chunk, unsigned int chunkSize) {
	/*UCHAR* hashValue = new UCHAR[SHA_DIGEST_LENGTH];
	SHA((UCHAR*)chunk, chunkSize, hashValue);
	return hashValue;*/
	return hashFunc.Hash(chunk, chunkSize);
}
