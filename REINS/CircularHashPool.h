#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"
class CircularHashPool : public VirtualHash
{
private:
	static const int POOL_SEGMENT_NUM = 2048;
	std::array<charPtMap, POOL_SEGMENT_NUM> mapPool;
	std::array<std::mutex, POOL_SEGMENT_NUM> mapPoolMutex;
	SelfMantainedCircularQueue circularQueue;
	std::mutex circularQueueMutex;

public:
	CircularHashPool(unsigned int size);
	~CircularHashPool();

	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);

	bool Find(unsigned char* hashValue);

	bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel);
};

