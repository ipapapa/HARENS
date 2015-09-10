#include "CircularHash.h"

CircularHash::CircularHash(unsigned int _size) : VirtualHash(_size), circularQueue(_size), map(size)
{
}

void CircularHash::SetupCircularHash(unsigned int _size) {
	SetupVirtualHash(_size);
	circularQueue.SetupCircularQueue(_size);
	map = std::unordered_map<unsigned long long, unsigned int>(_size);
}


CircularHash::~CircularHash()
{
	map.clear();
}

unsigned long long CircularHash::Add(const unsigned long long hashValue, const bool isDuplicated) {
	unsigned long long toBeDel = 0;
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
		map[hashValue] = 1;
		//map.insert({ hashValue, 1 });
	}
	return toBeDel;
}


bool CircularHash::Find(const unsigned long long hashValue) {
	return map.find(hashValue) != map.end();
}

bool CircularHash::FindAndAdd(const unsigned long long& hashValue, unsigned long long& toBeDel) {
	std::unordered_map<unsigned long long, unsigned int>::iterator it = map.find(hashValue);
	bool isFound = it != map.end();
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != NULL) {
		std::unordered_map<unsigned long long, unsigned int>::iterator toBeDelIt = map.find(toBeDel);
		if (toBeDelIt->second == 1) {
			map.erase(toBeDelIt, toBeDelIt);
		}
		else {
			toBeDelIt->second -= 1;
		}
	}
	if (isFound) {
		it->second += 1;
	}
	else {
		//map[hashValue] = 1;
		map.insert({ hashValue, 1 });
	}
	return isFound;
}