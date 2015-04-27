#include "CircularHashPool.h"


CircularHashPool::CircularHashPool(uint _size) : VirtualHash(_size)
{
	for (auto& segPool : mapPool)
		segPool = charPtMap(size / POOL_SEGMENT_NUM);
	for (auto& segQueue : circularQueuePool)
		segQueue = CircularQueue<uchar*>(size / POOL_SEGMENT_NUM);
}


CircularHashPool::~CircularHashPool()
{
	for (auto& segPool : mapPool)
		segPool.clear();
}


uchar* CircularHashPool::Add(uchar* hashValue) {
	uchar* to_be_del = NULL;
	int segNum = ((hashValue[0] << 8) + hashValue[1]) % POOL_SEGMENT_NUM;

	//Deal with the oldest hash value if the circular map is full
	circularQueuePoolLock[segNum].lock();
	to_be_del = circularQueuePool[segNum].Add(hashValue);
	circularQueuePoolLock[segNum].unlock();

	mapPoolLock[segNum].lock();
	if (to_be_del != NULL) {
		if (mapPool[segNum][to_be_del] == 1) {
			mapPool[segNum].erase(to_be_del);
		}
		else {
			mapPool[segNum][to_be_del] -= 1;
		}
	}
	if (mapPool[segNum].find(hashValue) == mapPool[segNum].end()) {
		mapPool[segNum].insert({ hashValue, 1 });
	}
	else {
		mapPool[segNum][hashValue] += 1;
	}
	mapPoolLock[segNum].unlock();
	return to_be_del;
}

bool CircularHashPool::Find(uchar* hashValue) {
	bool found;
	int segNum = ((hashValue[0] << 8) + hashValue[1]) % POOL_SEGMENT_NUM;
	mapPoolLock[segNum].lock();
	found = mapPool[segNum].find(hashValue) != mapPool[segNum].end();
	mapPoolLock[segNum].unlock();
	return found;
}
