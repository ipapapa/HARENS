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

unsigned int RedundancyEliminator_CPP_CollisionTest::
	fingerPrinting(deque<unsigned int> indexQ, char* package) {
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
		//if (circHash.Find(chunkHash)) { //find duplications
		//	duplicationSize += chunkLen;
		//	isDuplicate = true;
		//}
		//else {
		//	isDuplicate = false;
		//}
		//addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);
		/*unsigned long long toBeDel;
		if (circHash.FindAndAdd(chunkHash, toBeDel))
		duplicationSize += chunkLen;*/

		prevIdx = *iter;
	}
	return duplicationSize;
}