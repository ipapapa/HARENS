#pragma once
#include "Definition.h"
#include <type_traits>

/*
* A circular queue with LRU replacement strategy
*/
template<typename T>
class LRUQueue
{
private:
	//Generic functions that can automatically assign &toBeDel to illegal value
	template <typename S>
	void SetIllegal(S &toBeDel) {
		toBeDel = 0;
	}

	template <typename S>
	void SetIllegal(S *&toBeDel) {
		toBeDel = nullptr;
	}

	//Generic functions that can automatically free &toFree based on its value
	template <typename S>
	void Free(S &toFree) {
		//Do nothing
	}

	template <typename S>
	void Free(S *&toFree) {
		delete[] toFree;
	}

public:
	T* queue;			//A circular queue
	int front, rear;	//Rear point to the last used entry, there's an empty entry after rear
	unsigned int size;	//Size of the circular queue

	LRUQueue() {}

	LRUQueue(int _size) {
		SetupLRUQueue(_size);
	}

	void SetupLRUQueue(int _size) {
		size = _size;
		queue = new T[size];
		front = 0;
		rear = size - 1;
	}

	/*
	* Add a value into the queue, and return the value that is 
	* popped out according to the replacement strategy
	*/
	T Add(T hashValue) {
		T toBeDel;
		SetIllegal<T>(toBeDel);

		if ((rear + 2) % size == front) {
			toBeDel = queue[front];
			front = (front + 1) % size;
		}
		rear = (rear + 1) % size;
		queue[rear] = hashValue;
		return toBeDel;
	}

	~LRUQueue() {}
};