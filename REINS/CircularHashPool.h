#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"
class CircularHashPool : public VirtualHash
{
private:
	static const int POOL_SEGMENT_NUM = 2048;
	std::array<std::unordered_map<unsigned long long, unsigned int>, POOL_SEGMENT_NUM> mapPool;
	std::array<std::mutex, POOL_SEGMENT_NUM> mapPoolMutex;
	SelfMantainedCircularQueue circularQueue;
	std::mutex circularQueueMutex;

public:
	CircularHashPool(unsigned int size);
	~CircularHashPool();

	unsigned long long Add(const unsigned long long hashValue, const bool isDuplicated);

	bool Find(const unsigned long long hashValue);

	bool FindAndAdd(const unsigned long long& hashValue, unsigned long long& toBeDel);
};

