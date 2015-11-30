#pragma once
#include "LRUVirtualHash.h"
#include "LRUQueue.h"
#include "Definition.h"

/*
* Class for LRU hash that only deals with strings.
* It consists of a hash map and a LRU queue for LRU replacement.
* All the strings in LRUStrHash should be with the same length that defined in template.
*/
template <int str_len>
class LRUStrHash : public LRUVirtualHash<str_len> {
private:
	LRUQueue<unsigned char*> circularQueue;	//The LRU queue
	typename LRUVirtualHash<str_len>::charPtMap map;

public:
	LRUStrHash(): LRUVirtualHash<str_len>(0) {}
	LRUStrHash(unsigned int _size);
	void SetupLRUStrHash(unsigned int _size);
	~LRUStrHash();
	
	/*
	* Add a hash value knowing whether it is duplicated or not.
	* Return the obselete hash based on the LRU replacement policy,
	* if the LRU queue is full.
	*/
	unsigned char* Add(unsigned char* hashValue, const bool isDuplicated);

	/*
	* Find out if a hash value exists in the hash map.
	*/
	bool Find(unsigned char* hashValue);

	/*
	* Add a hash value without knowing whether it is duplicated or not.
	* Return the obselete hash as a reference based on the LRU replacement
	* policy, if the LRU queue is full.
	* Return if the hash value exists in the hash map or not.
	*/
	bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel);
};

template <int str_len>
LRUStrHash<str_len>::LRUStrHash(unsigned int _size)
	: LRUVirtualHash<str_len>(_size), circularQueue(_size), map(_size)
{
}

template <int str_len>
void LRUStrHash<str_len>::SetupLRUStrHash(unsigned int _size) {
	LRUStrHash<str_len>::SetupLRUVirtualHash(_size);
	circularQueue.SetupLRUQueue(_size);
	map = typename LRUVirtualHash<str_len>::charPtMap(_size);
}

template <int str_len>
LRUStrHash<str_len>::~LRUStrHash()
{
}

/*
* Add a hash value knowing whether it is duplicated or not.
* Return the obselete hash based on the LRU replacement policy,
* if the LRU queue is full.
*/
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
	}
	return toBeDel;
}

/*
* Find out if a hash value exists in the hash map.
*/
template <int str_len>
bool LRUStrHash<str_len>::Find(unsigned char* hashValue) {
	return map.find(hashValue) != map.end();
}

/*
* Add a hash value without knowing whether it is duplicated or not.
* Return the obselete hash as a reference based on the LRU replacement
* policy, if the LRU queue is full.
* Return if the hash value exists in the hash map or not.
*/
template <int str_len>
bool LRUStrHash<str_len>::FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel) {
	typename LRUVirtualHash<str_len>::charPtMap::iterator it = map.find(hashValue);
	bool isFound = it != map.end();
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != nullptr) {
		typename LRUVirtualHash<str_len>::charPtMap::iterator toBeDelIt = map.find(toBeDel);
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
		map[hashValue] = 1;
	}
	return isFound;
}
