#include "LRUHash_CollisionTest.h"

template <class T>
LRUHash_CollisionTest<T>::LRUHash_CollisionTest(unsigned int _size) 
	: circularQueue(_size), map(size)
{
	size = _size;
}

template <class T>
void LRUHash_CollisionTest<T>::SetupLRUHash_CollisionTest(unsigned int _size) {
	size = _size;
	circularQueue.SetupLRUQueue(_size);
	map = std::unordered_map<T, unsigned int>(_size);
}

template <class T>
LRUHash_CollisionTest<T>::~LRUHash_CollisionTest()
{
}

template <class T>
T LRUHash_CollisionTest<T>::Add(T hashValue, const bool isDuplicated) {
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
bool LRUHash_CollisionTest<T>::Find(T hashValue) {
	return map.find(hashValue) != map.end();
}

template <class T>
bool LRUHash_CollisionTest<T>::FindAndAdd(T hashValue, T* toBeDel){
	std::unordered_map<T, unsigned int>::iterator it = map.find(hashValue);
	bool isFound = it != map.end();
	toBeDel = circularQueue.Add(hashValue);
	if (toBeDel != 0) {
		std::unordered_map<T, unsigned int>::iterator toBeDelIt = map.find(toBeDel);
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