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
	ulong Add(ulong hashValue, bool isDuplicate);
	/*obsolete*/
	bool Find(ulong hashValue);

	bool FindAndAdd(const ulong& hashValue, ulong& toBeDel);
};

