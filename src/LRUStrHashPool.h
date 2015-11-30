#pragma once
#include "LRUVirtualHash.h"
#include "LRUQueue.h"
#include "Definition.h"

/*
* Class for a LRU hash that only deals with strings.
* The LRU hash is divided into segments - the pool.
* It is designed for faster access of large hash map.
* It consists of a pool of hash maps and LRU queues for LRU replacement.
* All the strings in LRUStrHashPool should be with the same length that defined in template.
*/
template <int str_len>
class LRUStrHashPool : public LRUVirtualHash<str_len>
{
private:
	static const int POOL_SEGMENT_NUM = 2048;	//Number of segments that the LRU hash is divided into
	std::array<typename LRUStrHash<str_len>::charPtMap, POOL_SEGMENT_NUM> mapPool;
	std::array<std::mutex, POOL_SEGMENT_NUM> mapPoolMutex;
	LRUQueue<unsigned char*> circularQueue;
	std::mutex circularQueueMutex;

public:
	LRUStrHashPool(unsigned int size);
	~LRUStrHashPool();
	
	/*
	* Add a hash value knowing whether it is duplicated or not.
	* Return the obselete hash based on the LRU replacement policy,
	* if the LRU queue is full.
	*/
	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);

	/*
	* Find out if a hash value exists in the hash map.
	*/
	bool Find(unsigned char* hashValue);

	/*
	* Add a hash value without knowing whether it is duplicated or not.
	* Return the obselete hash as a reference based on the LRU replacement
	* policy, if the LRU queue is full.
	* Return if the hash value exists in the hash map or not.
	*/
	bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel);
};

template <int str_len>
LRUStrHashPool<str_len>::LRUStrHashPool(unsigned int _size) : LRUVirtualHash<str_len>(_size)
{
	for (auto& segPool : mapPool)
		segPool = typename LRUStrHash<str_len>::charPtMap(_size / POOL_SEGMENT_NUM);
	circularQueue = LRUQueue<unsigned char*>(_size);
}

template <int str_len>
LRUStrHashPool<str_len>::~LRUStrHashPool()
{
	for (auto& segPool : mapPool)
		segPool.clear();
}

/*
* Add a hash value knowing whether it is duplicated or not.
* Return the obselete hash based on the LRU replacement policy,
* if the LRU queue is full.
*/
template <int str_len>
unsigned char* LRUStrHashPool<str_len>::Add(unsigned char* hashValue, const bool isDuplicated) {
	unsigned char* toBeDel;
	int segNum = (int)((hashValue[0] << 16) + (hashValue[1] << 8) + hashValue[2]) % POOL_SEGMENT_NUM;

	//Deal with the oldest hash value if the circular map is full
	circularQueueMutex.lock();
	toBeDel = circularQueue.Add(hashValue);
	circularQueueMutex.unlock();

	mapPoolMutex[segNum].lock();
	if (toBeDel != nullptr) {
		if (mapPool[segNum][toBeDel] == 1) {
			mapPool[segNum].erase(toBeDel);
		}
		else {
			mapPool[segNum][toBeDel] -= 1;
		}
	}
	if (isDuplicated) {
		//Use the newest char array as the key
		int occurence = mapPool[segNum][hashValue];
		mapPool[segNum].erase(hashValue);
		mapPool[segNum][hashValue] = occurence + 1;
	}
	else {
		mapPool[segNum].insert({ hashValue, 1 });
	}
	mapPoolMutex[segNum].unlock();
	return toBeDel;
}

/*
* Find out if a hash value exists in the hash map.
*/
template <int str_len>
bool LRUStrHashPool<str_len>::Find(unsigned char* hashValue) {
	bool isFound;
	int segNum = (int)((hashValue[0] << 16) + (hashValue[1] << 8) + hashValue[2]) % POOL_SEGMENT_NUM;
	mapPoolMutex[segNum].lock();
	isFound = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	mapPoolMutex[segNum].unlock();
	return isFound;
}

/*
* Add a hash value without knowing whether it is duplicated or not.
* Return the obselete hash as a reference based on the LRU replacement
* policy, if the LRU queue is full.
* Return if the hash value exists in the hash map or not.
*/
template <int str_len>
bool LRUStrHashPool<str_len>::FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel) {
	bool isFound;
	int segNum = (int)((hashValue[0] << 16) + (hashValue[1] << 8) + hashValue[2]) % POOL_SEGMENT_NUM;

	//Deal with the oldest hash value if the circular map is full
	circularQueueMutex.lock();
	toBeDel = circularQueue.Add(hashValue);
	circularQueueMutex.unlock();

	mapPoolMutex[segNum].lock();
	isFound = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	if (toBeDel != nullptr) {
		if (mapPool[segNum][toBeDel] == 1) {
			mapPool[segNum].erase(toBeDel);
		}
		else {
			mapPool[segNum][toBeDel] -= 1;
		}
	}
	if (isFound) {
		//Use the newest char array as the key
		int occurence = mapPool[segNum][hashValue];
		mapPool[segNum].erase(hashValue);
		mapPool[segNum].insert({ hashValue, occurence + 1 });
	}
	else {
		mapPool[segNum].insert({ hashValue, 1 });
	}
	mapPoolMutex[segNum].unlock();

	return isFound;
}