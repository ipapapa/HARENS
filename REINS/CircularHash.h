#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"

class CircularHash : public VirtualHash {
private:
	SelfMantainedCircularQueue<uchar*> circularQueue;
	charPtMap map;

public:
	CircularHash() {}
	CircularHash(uint _size);
	void SetupCircularHash(uint _size);
	~CircularHash();
	uchar* Add(uchar* hashValue);
	bool Find(uchar* hashValue);
};

