#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"
class CircularHashPool : public VirtualHash
{
private:
	static const int POOL_SEGMENT_NUM = 2048;
	std::array<std::unordered_map<ulong, uint>, POOL_SEGMENT_NUM> mapPool;
	std::array<std::mutex, POOL_SEGMENT_NUM> mapPoolMutex;
	SelfMantainedCircularQueue circularQueue;
	std::mutex circularQueueMutex;

public:
	CircularHashPool(uint size);
	~CircularHashPool();

	ulong Add(ulong hashValue, bool isDuplicate);

	bool Find(ulong hashValue);

	bool FindAndAdd(ulong& hashValue, ulong& toBeDel);
};

