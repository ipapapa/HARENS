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

ulong CircularHash::Add(const ulong hashValue, const bool isDuplicated) {
	ulong toBeDel = 0;
	//Deal with the oldest hash value if the circular map is full
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != NULL) {
		if (map[toBeDel] == 1) {
			map.erase(toBeDel);
		}
		else {
			map[toBeDel] -= 1;
		}
	}
	if (isDuplicated) {
		map[hashValue] += 1;
	}
	else {
		map.insert({ hashValue, 1 });
	}
	return toBeDel;
}


bool CircularHash::Find(const ulong hashValue) {
	return map.find(hashValue) != map.end();
}

bool CircularHash::FindAndAdd(const ulong& hashValue, ulong& toBeDel) {
	bool isFound = map.find(hashValue) != map.end();
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != NULL) {
		if (map[toBeDel] == 1) {
			map.erase(toBeDel);
		}
		else {
			map[toBeDel] -= 1;
		}
	}
	if (isFound) {
		map[hashValue] += 1;
	}
	else {
		map.insert({ hashValue, 1 });
	}
	return isFound;
}