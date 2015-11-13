#pragma once
#include "LRUVirtualHash.h"
#include "SelfMantainedLRUQueue.h"
#include "Definition.h"

//Generic class, T can be any basic types (no pointers or collections)
template <class T>
class LRUHash
{
private:
	unsigned int size;
	SelfMantainedLRUQueue<T> circularQueue;
	std::unordered_map<T, unsigned int> map;
public:
	LRUHash() {};

	LRUHash(unsigned int _size)
		: circularQueue(_size), map(size) {
		size = _size;
	}

	void SetupLRUHash(unsigned int _size);

	~LRUHash() {}

	T Add(T hashValue, const bool isDuplicated);

	bool Find(T hashValue);

	bool FindAndAdd(T hashValue, T* toBeDel);
};

template <class T>
void LRUHash<T>::SetupLRUHash(unsigned int _size) {
	size = _size;
	circularQueue.SetupLRUQueue(_size);
	map = std::unordered_map<T, unsigned int>(_size);
}

template <class T>
T LRUHash<T>::Add(T hashValue, const bool isDuplicated) {
	T toBeDel = 0;
	//Deal with the oldest hash value if the circular map is full
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != 0) {
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

template <class T>
bool LRUHash<T>::Find(T hashValue) {
	return map.find(hashValue) != map.end();
}

template <class T>
bool LRUHash<T>::FindAndAdd(T hashValue, T* toBeDel){
	typename std::unordered_map<T, unsigned int>::iterator it = map.find(hashValue);
	bool isFound = it != map.end();
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != 0) {
		typename std::unordered_map<T, unsigned int>::iterator toBeDelIt = map.find(toBeDel);
		if (toBeDelIt->second == 1) {
			map.erase(toBeDelIt, toBeDelIt);
		}
		else {
			toBeDelIt->second -= 1;
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