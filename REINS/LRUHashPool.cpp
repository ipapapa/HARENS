#include "LRUHashPool.h"
using namespace std;

LRUHashPool::LRUHashPool(unsigned int _size) : VirtualHash(_size)
{
	for (auto& segPool : mapPool)
		segPool = charPtMap(size / POOL_SEGMENT_NUM);
	circularQueue = SelfMantainedLRUQueue(size);
}


LRUHashPool::~LRUHashPool()
{
	for (auto& segPool : mapPool)
		segPool.clear();
}


unsigned char* LRUHashPool::Add(unsigned char* hashValue, const bool isDuplicated) {
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

bool LRUHashPool::Find(unsigned char* hashValue) {
	bool isFound;
	int segNum = (int)((hashValue[0] << 16) + (hashValue[1] << 8) + hashValue[2]) % POOL_SEGMENT_NUM;
	mapPoolMutex[segNum].lock();
	isFound = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	mapPoolMutex[segNum].unlock();
	return isFound;
}

bool LRUHashPool::FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel) {
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