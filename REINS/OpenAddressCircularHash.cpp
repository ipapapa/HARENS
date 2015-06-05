#include "OpenAddressCircularHash.h"

OpenAddressCircularHash::OpenAddressCircularHash(uint _size) : VirtualHash(_size), circularQueue(_size)
{
}

void OpenAddressCircularHash::SetupCircularHash(uint _size) {
	SetupVirtualHash(_size);
	circularQueue.SetupCircularQueue(_size);
}


OpenAddressCircularHash::~OpenAddressCircularHash()
{
}

/*obsolete*/
ulong OpenAddressCircularHash::Add(ulong hashValue, bool isDuplicate){ return 0; }
/*obsolete*/
bool OpenAddressCircularHash::Find(ulong hashValue){ return false; }

bool OpenAddressCircularHash::FindAndAdd(const ulong& hashValue, ulong& toBeDel) {
	int location = map.Find(hashValue);
	bool insertSucceed = true;
	if (location == -1) {
		insertSucceed = map.InsertNew(hashValue);
	}
	else {
		map.vals[location] += 1;
	}

	if (insertSucceed) {
		toBeDel = circularQueue.Add(hashValue);
		map.Reduce(toBeDel);
	}
	else
		toBeDel = 0;

	return (location != -1);
}