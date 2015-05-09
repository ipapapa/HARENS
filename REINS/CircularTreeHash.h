#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"
#include "TreeHash.h"

class CircularTreeHash : public VirtualHash {
private:
	SelfMantainedCircularQueue circularQueue;
	TreeHash map;

public:
	CircularTreeHash() {}
	CircularTreeHash(uint _size);
	void SetupCircularHash(uint _size);
	~CircularTreeHash();
	ulong Add(ulong hashValue, bool isDuplicate);
	bool Find(ulong hashValue);
	bool FindAndAdd(ulong& hashValue, ulong& toBeDel);
};

