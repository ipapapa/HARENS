#pragma once
#include "VirtualHash.h"
#include "SelfMantainedLRUQueue.h"
#include "Definition.h"
class LRUHashPool : public VirtualHash
{
private:
	static const int POOL_SEGMENT_NUM = 2048;
	std::array<charPtMap, POOL_SEGMENT_NUM> mapPool;
	std::array<std::mutex, POOL_SEGMENT_NUM> mapPoolMutex;
	SelfMantainedLRUQueue<unsigned char*> circularQueue;
	std::mutex circularQueueMutex;

public:
	LRUHashPool(unsigned int size);
	~LRUHashPool();

	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);

	bool Find(unsigned char* hashValue);

	bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel);
};

