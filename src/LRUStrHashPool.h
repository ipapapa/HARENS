#pragma once
#include "LRUVirtualHash.h"
#include "LRUQueue.h"
#include "Definition.h"

template <int str_len>
class LRUStrHashPool : public LRUVirtualHash<str_len>
{
private:
	static const int POOL_SEGMENT_NUM = 2048;
	std::array<typename LRUStrHash<str_len>::charPtMap, POOL_SEGMENT_NUM> mapPool;
	std::array<std::mutex, POOL_SEGMENT_NUM> mapPoolMutex;
	LRUQueue<unsigned char*> circularQueue;
	std::mutex circularQueueMutex;

public:
	LRUStrHashPool(unsigned int size);
	~LRUStrHashPool();

	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);

	bool Find(unsigned char* hashValue);

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

template <int str_len>
bool LRUStrHashPool<str_len>::Find(unsigned char* hashValue) {
	bool isFound;
	int segNum = (int)((hashValue[0] << 16) + (hashValue[1] << 8) + hashValue[2]) % POOL_SEGMENT_NUM;
	mapPoolMutex[segNum].lock();
	isFound = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	mapPoolMutex[segNum].unlock();
	return isFound;
}

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