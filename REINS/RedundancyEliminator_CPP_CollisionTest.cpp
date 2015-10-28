#include "RedundancyEliminator_CPP_CollisionTest.h"


RedundancyEliminator_CPP_CollisionTest::RedundancyEliminator_CPP_CollisionTest()
{
}


RedundancyEliminator_CPP_CollisionTest::~RedundancyEliminator_CPP_CollisionTest()
{
}

void RedundancyEliminator_CPP_CollisionTest::SetupRedundancyEliminator_CPP_CollisionTest() {
	SetupRedundancyEliminator_CPP();
	circHash.SetupLRUHash_CollisionTest(MAX_CHUNK_NUM);
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
	RabinFingerPrinting(deque<unsigned int> indexQ, char* package) {
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

unsigned int RedundancyEliminator_CPP_CollisionTest::
	SHA1FingerPrinting(deque<unsigned int> indexQ, char* package) {
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
		RedundancyEliminator_CPP::computeChunkHash(chunk, chunkLen, chunkHash);
		if (RedundancyEliminator_CPP::circHash.Find(chunkHash)) { //find duplications
			duplicationSize += chunkLen;
			isDuplicate = true;
		}
		else {
			isDuplicate = false;
		}
		RedundancyEliminator_CPP::addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);
		/*unsigned long long toBeDel;
		if (circHash.FindAndAdd(chunkHash, toBeDel))
		duplicationSize += chunkLen;*/

		prevIdx = *iter;
	}
	return duplicationSize;
}

tuple<unsigned int, unsigned int> RedundancyEliminator_CPP_CollisionTest::
	RabinFingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package) {
	unsigned int duplicationSize = 0;
	unsigned int falseReportSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	bool isDuplicate;
	bool isFalseReport;
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
		
		isFalseReport = false;
		if (isDuplicate) {
			tuple<char*, int> oldChunk = rabinMap[chunkHash];
			char* oldChunkContent = get<0>(oldChunk);
			int oldChunkLen = get<1>(oldChunk);
			if (oldChunkLen != chunkLen) {
				isFalseReport = true;
			}
			else {
				for (int i = 0; i < chunkLen; ++i) {
					if (oldChunkContent[i] != chunk[i]) {
						isFalseReport = true;
						break;
					}
				}
			}
		}
		else {
			char* chunkContent = new char[chunkLen];
			memcpy(chunkContent, chunk, chunkLen);
			rabinMap[chunkHash] = make_tuple(chunkContent, chunkLen);
		}
		if (isFalseReport)
			falseReportSize += chunkLen;

		addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);

		prevIdx = *iter;
	}
	return make_tuple(duplicationSize, falseReportSize);
}

tuple<unsigned int, unsigned int> RedundancyEliminator_CPP_CollisionTest::
	SHA1FingerPrintingWithCollisionCheck(deque<unsigned int> indexQ, char* package) {
	unsigned int duplicationSize = 0;
	unsigned int falseReportSize = 0;
	unsigned int prevIdx = 0;
	char* chunk;
	bool isDuplicate;
	bool isFalseReport;
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
		RedundancyEliminator_CPP::computeChunkHash(chunk, chunkLen, chunkHash);
		if (RedundancyEliminator_CPP::circHash.Find(chunkHash)) { //find duplications
			duplicationSize += chunkLen;
			isDuplicate = true;
		}
		else {
			isDuplicate = false;
		}

		isFalseReport = false;
		if (isDuplicate) {
			tuple<char*, int> oldChunk = sha1Map[chunkHash];
			char* oldChunkContent = get<0>(oldChunk);
			int oldChunkLen = get<1>(oldChunk);
			if (oldChunkLen != chunkLen) {
				isFalseReport = true;
			}
			else {
				for (int i = 0; i < chunkLen; ++i) {
					if (oldChunkContent[i] != chunk[i]) {
						isFalseReport = true;
						break;
					}
				}
			}
		}
		else {
			char* chunkContent = new char[chunkLen];
			memcpy(chunkContent, chunk, chunkLen);
			sha1Map[chunkHash] = make_tuple(chunkContent, chunkLen);
		}
		if (isFalseReport)
			falseReportSize += chunkLen;

		RedundancyEliminator_CPP::addNewChunk(chunkHash, chunk, chunkLen, isDuplicate);
		/*unsigned long long toBeDel;
		if (circHash.FindAndAdd(chunkHash, toBeDel))
		duplicationSize += chunkLen;*/

		prevIdx = *iter;
	}
	return make_tuple(duplicationSize, falseReportSize);
}

inline unsigned long long RedundancyEliminator_CPP_CollisionTest::computeChunkHash
	(char* chunk, unsigned int chunkSize) {
	return hashFunc.Hash(chunk, chunkSize);
}