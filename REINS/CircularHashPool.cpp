#include "CircularHashPool.h"
using namespace std;

CircularHashPool::CircularHashPool(uint _size) : VirtualHash(_size)
{
	for (auto& segPool : mapPool)
		segPool = std::unordered_map<ulong, uint>(size / POOL_SEGMENT_NUM);
	circularQueue = SelfMantainedCircularQueue(size);
}


CircularHashPool::~CircularHashPool()
{
	for (auto& segPool : mapPool)
		segPool.clear();
}


ulong CircularHashPool::Add(const ulong hashValue, const bool isDuplicated) {
	ulong toBeDel = 0;
	int segNum = hashValue % POOL_SEGMENT_NUM;

	//Deal with the oldest hash value if the circular map is full
	circularQueueMutex.lock();
	toBeDel = circularQueue.Add(hashValue);
	circularQueueMutex.unlock();

	mapPoolMutex[segNum].lock();
	if (toBeDel != 0) {
		if (mapPool[segNum][toBeDel] == 1) {
			mapPool[segNum].erase(toBeDel);
		}
		else {
			mapPool[segNum][toBeDel] -= 1;
		}
	}
	if (isDuplicated) {
		mapPool[segNum][hashValue] += 1;
	}
	else {
		mapPool[segNum].insert({ hashValue, 1 });
	}
	mapPoolMutex[segNum].unlock();
	return toBeDel;
}

bool CircularHashPool::Find(const ulong hashValue) {
	bool isFound;
	int segNum = hashValue % POOL_SEGMENT_NUM;
	mapPoolMutex[segNum].lock();
	isFound = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	mapPoolMutex[segNum].unlock();
	return isFound;
}

bool CircularHashPool::FindAndAdd(const ulong& hashValue, ulong& toBeDel) {
	bool isFound;
	int segNum = hashValue % POOL_SEGMENT_NUM;

	//Deal with the oldest hash value if the circular map is full
	circularQueueMutex.lock();
	toBeDel = circularQueue.Add(hashValue);
	circularQueueMutex.unlock();

	mapPoolMutex[segNum].lock();
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
	mapPoolMutex[segNum].unlock();

	return isFound;
}