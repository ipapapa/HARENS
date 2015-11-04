#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"
#include "TreeHash.h"
#include "QuadTree.h"

class OpenAddressCircularHash : public VirtualHash {
private:
	SelfMantainedCircularQueue circularQueue;
	TreeHash map;

public:
	OpenAddressCircularHash() {}
	OpenAddressCircularHash(unsigned int _size);
	void SetupCircularHash(unsigned int _size);
	~OpenAddressCircularHash();

	/*obsolete*/
	unsigned long long Add(const unsigned long long hashValue, const bool isDuplicated);
	/*obsolete*/
	bool Find(const unsigned long long hashValue);

	bool FindAndAdd(const unsigned long long& hashValue, unsigned long long& toBeDel);
};

