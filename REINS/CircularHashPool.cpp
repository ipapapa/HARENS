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
	ulong to_be_del = 0;
	int segNum = hashValue % POOL_SEGMENT_NUM;

	//Deal with the oldest hash value if the circular map is full
	circularQueuePoolLock[segNum].lock();
	to_be_del = circularQueuePool[segNum].Add(hashValue);
	circularQueuePoolLock[segNum].unlock();

	mapPoolLock[segNum].lock();
	if (to_be_del != 0) {
		if (mapPool[segNum][to_be_del] == 1) {
			mapPool[segNum].erase(to_be_del);
		}
		else {
			mapPool[segNum][to_be_del] -= 1;
		}
	}
	if (isDuplicate) {
		mapPool[segNum][hashValue] += 1;
	}
	else {
		mapPool[segNum].insert({ hashValue, 1 });
	}
	mapPoolLock[segNum].unlock();
	return to_be_del;
}

bool CircularHashPool::Find(ulong hashValue) {
	bool found;
	int segNum = hashValue % POOL_SEGMENT_NUM;
	mapPoolLock[segNum].lock();
	found = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	mapPoolLock[segNum].unlock();
	return found;
}
