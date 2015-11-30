#pragma once
#include "LRUVirtualHash.h"
#include "LRUQueue.h"
#include "Definition.h"

/*
* Generic class for LRU hash, T can be any basic types (no pointers or collections).
* It consists of a hash map and a LRU queue for LRU replacement.
*/
template <class T>
class LRUHash
{
private:
	LRUQueue<T> circularQueue;	//The LRU queue
	std::unordered_map<T, unsigned int> map;
public:
	LRUHash() {};

	LRUHash(unsigned int _size)
		: circularQueue(_size), map(_size) {}

	void SetupLRUHash(unsigned int _size);

	~LRUHash() {}

	/*
	* Add a hash value knowing whether it is duplicated or not.
	* Return the obselete hash based on the LRU replacement policy,
	* if the LRU queue is full.
	*/
	T Add(T hashValue, const bool isDuplicated);

	/*
	* Find out if a hash value exists in the hash map.
	*/
	bool Find(T hashValue);

	/*
	* Add a hash value without knowing whether it is duplicated or not.
	* Return the obselete hash as a reference based on the LRU replacement
	* policy, if the LRU queue is full.
	* Return if the hash value exists in the hash map or not.
	*/
	bool FindAndAdd(T hashValue, T* toBeDel);
};

template <class T>
void LRUHash<T>::SetupLRUHash(unsigned int _size) {
	circularQueue.SetupLRUQueue(_size);
	map = std::unordered_map<T, unsigned int>(_size);
}

/*
* Add a hash value knowing whether it is duplicated or not.
* Return the obselete hash based on the LRU replacement policy,
* if the LRU queue is full.
*/
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

/*
* Find out if a hash value exists in the hash map.
*/
template <class T>
bool LRUHash<T>::Find(T hashValue) {
	return map.find(hashValue) != map.end();
}

/*
* Add a hash value without knowing whether it is duplicated or not.
* Return the obselete hash as a reference based on the LRU replacement
* policy, if the LRU queue is full.
* Return if the hash value exists in the hash map or not.
*/
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