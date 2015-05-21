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
void RedundancyEliminator_CPP::addNewChunk(ulong hashValue, char* chunk, uint chunkSize, bool isDuplicate) {
	ulong to_be_del = circHash.Add(hashValue, isDuplicate);
	//fstream file(hashValue.c_str(), std::fstream::in|std::fstream::out);
	////we are actually supposed to do something with chunkSize here
	//file << chunk;
	//file.close();
}

deque<uint> RedundancyEliminator_CPP::chunking(char* package, uint packageSize) {
	deque<uint> indexQ = deque<uint>();
	char* chunk = new char[WINDOW_SIZE];

	for (uint i = 0; i < packageSize - WINDOW_SIZE + 1; ++i) {
		memcpy(chunk, &(package[i]), WINDOW_SIZE);
		ulong windowFingerPrint = hashFunc.Hash(chunk, WINDOW_SIZE);
		if ((windowFingerPrint & P_MINUS) == 0) { // marker found
			indexQ.push_back(i);
		}
	}
	return indexQ;
}

uint RedundancyEliminator_CPP::fingerPrinting(deque<uint> indexQ, char* package) {
	uint duplicationSize = 0;
	uint prevIdx = 0;
	char* chunk;
	bool isDuplicate;
	for (deque<uint>::const_iterator iter = indexQ.begin(); iter != indexQ.end(); ++iter) {
		if (prevIdx == 0) {
			prevIdx = *iter;
			continue;
		}
		uint chunkLen = *iter - prevIdx;
		chunk = &(package[prevIdx]);
		ulong chunkHash = computeChunkHash(chunk, chunkLen);
		if (circHash.Find(chunkHash)) { //find duplications
			duplicationSize += chunkLen;
			isDuplicate = true;
		}
		else {
			isDuplicate = false;
		}
		addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);

		prevIdx = *iter;
	}
	return duplicationSize;
}

uint RedundancyEliminator_CPP::eliminateRedundancy(char* package, uint packageSize) {
	clock_t start = clock();
	deque<uint> indexQ = chunking(package, packageSize);
	double time = (clock() - start) * 1000 / CLOCKS_PER_SEC;
	printf("Chunk partition time: %f ms\n", time);
	return fingerPrinting(indexQ, package);
}

/*
Compute the hash value of chunk, should use sha3 to avoid collision,
I'm using rabin hash here for convience
*/
inline ulong RedundancyEliminator_CPP::computeChunkHash(char* chunk, uint chunkSize) {
	/*uchar* hashValue = new uchar[SHA_DIGEST_LENGTH];
	SHA((uchar*)chunk, chunkSize, hashValue);
	return hashValue;*/
	return hashFunc.Hash(chunk, chunkSize);
}
