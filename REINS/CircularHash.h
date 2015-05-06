#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"

class CircularHash : public VirtualHash {
private:
	SelfMantainedCircularQueue circularQueue;
	std::unordered_map<ulong, uint> map;

public:
	CircularHash() {}
	CircularHash(uint _size);
	void SetupCircularHash(uint _size);
	~CircularHash();
	ulong Add(ulong hashValue, bool isDuplicate);
	bool Find(ulong hashValue);
	bool FindAndAdd(ulong& hashValue, ulong& toBeDel);
};

