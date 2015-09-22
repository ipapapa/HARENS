#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"

class CircularHash : public VirtualHash {
private:
	SelfMantainedCircularQueue circularQueue;
	charPtMap map;

public:
	CircularHash(): VirtualHash(0) {}
	CircularHash(unsigned int _size);
	void SetupCircularHash(unsigned int _size);
	~CircularHash();
	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);
	bool Find(unsigned char* hashValue);
	bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel);
};

