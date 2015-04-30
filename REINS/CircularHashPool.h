#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"
class CircularHashPool : public VirtualHash
{
private:
	static const int POOL_SEGMENT_NUM = 2048;
	std::array<charPtMap, POOL_SEGMENT_NUM> mapPool;
	std::array<std::mutex, POOL_SEGMENT_NUM> mapPoolLock;
	std::array<SelfMantainedCircularQueue, POOL_SEGMENT_NUM> circularQueuePool;
	std::array<std::mutex, POOL_SEGMENT_NUM> circularQueuePoolLock;

public:
	CircularHashPool(uint size);
	~CircularHashPool();

	uchar* Add(uchar* hashValue);

	bool Find(uchar* hashValue);
};

