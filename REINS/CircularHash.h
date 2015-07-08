#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"

class CircularHash : public VirtualHash {
private:
	SelfMantainedCircularQueue circularQueue;
	std::unordered_map<unsigned long long, unsigned int> map;

public:
	CircularHash() {}
	CircularHash(unsigned int _size);
	void SetupCircularHash(unsigned int _size);
	~CircularHash();
	unsigned long long Add(const unsigned long long hashValue, const bool isDuplicated);
	bool Find(const unsigned long long hashValue);
	bool FindAndAdd(const unsigned long long& hashValue, unsigned long long& toBeDel);
};

