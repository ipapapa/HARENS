#pragma once
#include "LRUVirtualHash.h"
#include "SelfMantainedLRUQueue.h"
#include "Definition.h"

template <int str_len>
class LRUStrHash : public LRUVirtualHash<str_len> {
private:
	SelfMantainedLRUQueue<unsigned char*> circularQueue;
	charPtMap map;

public:
	LRUStrHash(): LRUVirtualHash(0) {}
	LRUStrHash(unsigned int _size);
	void SetupLRUStrHash(unsigned int _size);
	~LRUStrHash();
	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);
	bool Find(unsigned char* hashValue);
	bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel);
};

template <int str_len>
LRUStrHash<str_len>::LRUStrHash<str_len>(unsigned int _size) : LRUVirtualHash<str_len>(_size), circularQueue(_size), map(size)
{
}

template <int str_len>
void LRUStrHash<str_len>::SetupLRUStrHash(unsigned int _size) {
	SetupLRUVirtualHash(_size);
	circularQueue.SetupLRUQueue(_size);
	map = charPtMap(_size);
}

template <int str_len>
LRUStrHash<str_len>::~LRUStrHash()
{
}

template <int str_len>
unsigned char* LRUStrHash<str_len>::Add(unsigned char* hashValue, const bool isDuplicated) {
	unsigned char* toBeDel;
	//Deal with the oldest hash value if the circular map is full
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != nullptr) {
		if (map[toBeDel] == 1) {
			map.erase(toBeDel);
		}
		else {
			map[toBeDel] -= 1;
		}
	}
	if (isDuplicated) {
		//Use the newest char array as the key
		int occurence = map[hashValue];
		map.erase(hashValue);
		map[hashValue] = occurence + 1;
	}
	else {
		map[hashValue] = 1;
		//map.insert({ hashValue, 1 });
	}
	return toBeDel;
}

template <int str_len>
bool LRUStrHash<str_len>::Find(unsigned char* hashValue) {
	return map.find(hashValue) != map.end();
}

template <int str_len>
bool LRUStrHash<str_len>::FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel) {
	typename charPtMap::iterator it = map.find(hashValue);
	bool isFound = it != map.end();
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != nullptr) {
		typename charPtMap::iterator toBeDelIt = map.find(toBeDel);
		if (toBeDelIt->second == 1) {
			map.erase(toBeDelIt, toBeDelIt);
		}
		else {
			toBeDelIt->second -= 1;
		}
	}
	if (isFound) {
		//Use the newest char array as the key
		int occurence = it->second;
		map.erase(it, it);
		map[hashValue] = occurence + 1;
	}
	else {
		//map[hashValue] = 1;
		map.insert({ hashValue, 1 });
	}
	return isFound;
}
