#include "CircularHash.h"

CircularHash::CircularHash(uint _size) : VirtualHash(_size), circularQueue(_size), map(size)
{
}

void CircularHash::SetupCircularHash(uint _size) {
	SetupVirtualHash(_size);
	circularQueue.SetupCircularQueue(_size);
	map = charPtMap(_size);
}


CircularHash::~CircularHash()
{
	map.clear();
}

uchar* CircularHash::Add(uchar* hashValue) {
	uchar* to_be_del = NULL;
	//Deal with the oldest hash value if the circular map is full
	to_be_del = circularQueue.Add(hashValue);
	if (to_be_del != NULL) {
		if (map[to_be_del] == 1) {
			map.erase(to_be_del);
		}
		else {
			map[to_be_del] -= 1;
		}
	}
	if (map.find(hashValue) == map.end()) {
		map.insert({ hashValue, 1 });
	}
	else {
		map[hashValue] += 1;
	}
	return to_be_del;
}


bool CircularHash::Find(uchar* hashValue) {
	return map.find(hashValue) != map.end();
}