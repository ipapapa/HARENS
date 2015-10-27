#pragma once
#include "VirtualHash.h"
#include "SelfMantainedLRUQueue.h"
#include "Definition.h"

class LRUHash : public VirtualHash {
private:
	SelfMantainedLRUQueue circularQueue;
	charPtMap map;

public:
	LRUHash(): VirtualHash(0) {}
	LRUHash(unsigned int _size);
	void SetupLRUHash(unsigned int _size);
	~LRUHash();
	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);
	bool Find(unsigned char* hashValue);
	bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel);
};

