#include "RedundancyEliminator_CPP_CollisionTest.h"


RedundancyEliminator_CPP_CollisionTest::RedundancyEliminator_CPP_CollisionTest()
{
}


RedundancyEliminator_CPP_CollisionTest::~RedundancyEliminator_CPP_CollisionTest()
{
}

inline unsigned long long RedundancyEliminator_CPP_CollisionTest::
	computeChunkHash(char* chunk, unsigned int chunkSize) {
	return hashFunc.Hash(chunk, chunkSize);
}

/*
Add a new chunck into the file system, if the hash value queue is full, also delete the oldest chunk.
*/
void RedundancyEliminator_CPP_CollisionTest::addNewChunk(unsigned long long hashValue
	, char* chunk, unsigned int chunkSize, bool isDuplicate) {
	unsigned long long toBeDel = circHash.Add(hashValue, isDuplicate);
	//Remove chunk corresponding to toBeDel from storage
	//fstream file(hashValue.c_str(), std::fstream::in|std::fstream::out);
	////we are actually supposed to do something with chunkSize here
	//file << chunk;
	//file.close();
}

unsigned int RedundancyEliminator_CPP_CollisionTest::
	RabinFingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package) {
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

		prevIdx = *iter;
	}
	return duplicationSize;
}

unsigned int RedundancyEliminator_CPP::
	SHA1FingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package) {
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
		unsigned char* chunkHash = new unsigned char[SHA_DIGEST_LENGTH];
		computeChunkHash(chunk, chunkLen, chunkHash);
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

inline unsigned long long RedundancyEliminator_CPP_CollisionTest::computeChunkHash
	(char* chunk, unsigned int chunkSize) {
	return hashFunc.Hash(chunk, chunkSize);
}