#pragma once
#include "VirtualHash.h"
#include "SelfMantainedCircularQueue.h"
#include "Definition.h"
#include "TreeHash.h"

class OpenAddressCircularHash : public VirtualHash {
private:
	SelfMantainedCircularQueue circularQueue;
	TreeHash map;

public:
	OpenAddressCircularHash() {}
	OpenAddressCircularHash(uint _size);
	void SetupCircularHash(uint _size);
	~OpenAddressCircularHash();

	/*obsolete*/
	ulong Add(const ulong hashValue, const bool isDuplicated);
	/*obsolete*/
	bool Find(const ulong hashValue);

	bool FindAndAdd(const ulong& hashValue, ulong& toBeDel);
};

