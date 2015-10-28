#pragma once
#include "VirtualHash.h"
#include "SelfMantainedLRUQueue.h"
#include "Definition.h"

//Generic class, T can be any basic types (no pointers or collections)
template <class T>
class LRUHash_CollisionTest
{
private:
	unsigned int size;
	SelfMantainedLRUQueue<T> circularQueue;
	std::unordered_map<T, unsigned int> map;
public:
	LRUHash_CollisionTest() : VirtualHash(0) {};
	LRUHash_CollisionTest(unsigned int _size);
	void SetupLRUHash_CollisionTest(unsigned int _size);
	~LRUHash_CollisionTest();
	T Add(T hashValue, const bool isDuplicated);
	bool Find(T hashValue);
	bool FindAndAdd(T hashValue, T* toBeDel);
};

