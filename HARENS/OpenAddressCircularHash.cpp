#include "OpenAddressCircularHash.h"

OpenAddressCircularHash::OpenAddressCircularHash(unsigned int _size) : VirtualHash(_size), circularQueue(_size)
{
}

void OpenAddressCircularHash::SetupCircularHash(unsigned int _size) {
	SetupVirtualHash(_size);
	circularQueue.SetupCircularQueue(_size);
}


OpenAddressCircularHash::~OpenAddressCircularHash()
{
}

/*obsolete*/
unsigned long long OpenAddressCircularHash::Add(const unsigned long long hashValue, const bool isDuplicated){ return 0; }
/*obsolete*/
bool OpenAddressCircularHash::Find(const unsigned long long hashValue){ return false; }

bool OpenAddressCircularHash::FindAndAdd(const unsigned long long& hashValue, unsigned long long& toBeDel) {
	/*bool found = map.FindAndInsert(hashValue);
	toBeDel = circularQueue.Add(hashValue);
	map.Erase(toBeDel);
	return found;*/

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