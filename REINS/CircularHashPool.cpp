#include "CircularHashPool.h"


CircularHashPool::CircularHashPool(uint _size) : VirtualHash(_size)
{
	for (auto& segPool : mapPool)
		segPool = std::unordered_map<ulong, uint>(size / POOL_SEGMENT_NUM);
	for (auto& segQueue : circularQueuePool)
		segQueue = SelfMantainedCircularQueue(size / POOL_SEGMENT_NUM);
}


CircularHashPool::~CircularHashPool()
{
	for (auto& segPool : mapPool)
		segPool.clear();
}


ulong CircularHashPool::Add(ulong hashValue, bool isDuplicate) {
	ulong toBeDel = 0;
	int segNum = hashValue % POOL_SEGMENT_NUM;

	//Deal with the oldest hash value if the circular map is full
	circularQueuePoolLock[segNum].lock();
	toBeDel = circularQueuePool[segNum].Add(hashValue);
	circularQueuePoolLock[segNum].unlock();

	mapPoolLock[segNum].lock();
	if (toBeDel != 0) {
		if (mapPool[segNum][toBeDel] == 1) {
			mapPool[segNum].erase(toBeDel);
		}
		else {
			mapPool[segNum][toBeDel] -= 1;
		}
	}
	if (isDuplicate) {
		mapPool[segNum][hashValue] += 1;
	}
	else {
		mapPool[segNum].insert({ hashValue, 1 });
	}
	mapPoolLock[segNum].unlock();
	return toBeDel;
}

bool CircularHashPool::Find(ulong hashValue) {
	bool isFound;
	int segNum = hashValue % POOL_SEGMENT_NUM;
	mapPoolLock[segNum].lock();
	isFound = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	mapPoolLock[segNum].unlock();
	return isFound;
}

bool CircularHashPool::FindAndAdd(ulong& hashValue, ulong& toBeDel) {
	bool isFound;
	int segNum = hashValue % POOL_SEGMENT_NUM;

	toBeDel = circularQueuePool[segNum].Add(hashValue);
	//Deal with the oldest hash value if the circular map is full
	circularQueuePoolLock[segNum].lock();
	toBeDel = circularQueuePool[segNum].Add(hashValue);
	circularQueuePoolLock[segNum].unlock();

	mapPoolLock[segNum].lock();
	isFound = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	if (toBeDel != 0) {
		if (mapPool[segNum][toBeDel] == 1) {
			mapPool[segNum].erase(toBeDel);
		}
		else {
			mapPool[segNum][toBeDel] -= 1;
		}
	}
	if (isFound) {
		mapPool[segNum][hashValue] += 1;
	}
	else {
		mapPool[segNum].insert({ hashValue, 1 });
	}
	mapPoolLock[segNum].unlock();

	return isFound;
}