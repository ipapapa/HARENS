#pragma once
#include "RedundancyEliminator_CPP.h"
#include "LRUHash.h"

/*
* The redundancy elimination module for hash collision tests
*/
class RedundancyEliminator_CPP_CollisionTest: public RedundancyEliminator_CPP
{
private:
	//The LRU hashes for three kind of hash functions
	LRUHash<unsigned long long> rabinLRUHash;
	LRUStrHash<SHA1_HASH_LENGTH> sha1LRUHash;
	LRUStrHash<MD5_DIGEST_LENGTH> md5LRUHash;

	/*
	The hash maps that store all the duplications detected 
	by using three kind of hash functions
	*/
	std::unordered_map<unsigned long long, 
					   std::tuple<char*, int>> 
					   rabinMap;
	std::unordered_map<unsigned char*, 
					   std::tuple<char*, int>, 
					   CharArrayHashFunc<SHA1_HASH_LENGTH>, 
					   CharArrayEqualTo<SHA1_HASH_LENGTH>> 
					   sha1Map;
	std::unordered_map<unsigned char*,
					   std::tuple<char*, int>,
					   CharArrayHashFunc<MD5_DIGEST_LENGTH>,
					   CharArrayEqualTo<MD5_DIGEST_LENGTH>>
					   md5Map;

	/*
	The addNewChunk function implemented for the algorithm using 
	three kind of hash functions repectively
	*/
	inline void addNewChunkRabin(unsigned long long hashValue, 
								 char* chunk, 
								 unsigned int chunkSize, 
								 bool isDuplicate);

	inline void addNewChunkSha1(unsigned char* hashValue,
								char* chunk,
								unsigned int chunkSize,
								bool isDuplicate);

	inline void addNewChunkMd5(unsigned char* hashValue,
							   char* chunk,
							   unsigned int chunkSize,
							   bool isDuplicate);
public:
	RedundancyEliminator_CPP_CollisionTest();
	void SetupRedundancyEliminator_CPP_CollisionTest();
	~RedundancyEliminator_CPP_CollisionTest();

	inline unsigned long long ComputeRabinHash(char* chunk, unsigned int chunkSize);
	unsigned int RabinFingerPrinting(std::deque<unsigned int> indexQ, char* package);
	unsigned int Sha1FingerPrinting(std::deque<unsigned int> indexQ, char* package);
	unsigned int Md5FingerPrinting(std::deque<unsigned int> indexQ, char* package);
	std::tuple<unsigned int, unsigned int> 
		RabinFingerPrintingWithCollisionCheck(std::deque<unsigned int> 
											  indexQ, char* package);
	std::tuple<unsigned int, unsigned int>
		Sha1FingerPrintingWithCollisionCheck(std::deque<unsigned int> indexQ,
											 char* package);
	std::tuple<unsigned int, unsigned int>
		Md5FingerPrintingWithCollisionCheck(std::deque<unsigned int> indexQ,
											char* package);
};