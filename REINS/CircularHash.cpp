#include "CircularHash.h"

CircularHash::CircularHash(uint _size) : VirtualHash(_size), circularQueue(_size), map(size)
{
}

void CircularHash::SetupCircularHash(uint _size) {
	SetupVirtualHash(_size);
	circularQueue.SetupCircularQueue(_size);
	map = std::unordered_map<ulong, uint>(_size);
}


CircularHash::~CircularHash()
{
	map.clear();
}

ulong CircularHash::Add(ulong hashValue, bool isDuplicate) {
	ulong to_be_del = 0;
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
	if (isDuplicate) {
		map[hashValue] += 1;
	}
	else {
		map.insert({ hashValue, 1 });
	}
	return to_be_del;
}


bool CircularHash::Find(ulong hashValue) {
	return map.find(hashValue) != map.end();
}