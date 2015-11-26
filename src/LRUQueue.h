#pragma once
#include "Definition.h"
#include <type_traits>

template<typename T>
class LRUQueue
{
private:
	template <typename S>
	void SetIllegal(S &toBeDel) {
		toBeDel = 0;
	}

	template <typename S>
	void SetIllegal(S *&toBeDel) {
		toBeDel = nullptr;
	}

	template <typename S>
	void Free(S &toFree) {
		//Do nothing
	}

	template <typename S>
	void Free(S *&toFree) {
		delete[] toFree;
	}

public:
	T* queue;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	unsigned int size;

	LRUQueue() {}

	LRUQueue(int _size) {
		SetupLRUQueue(_size);
	}

	void SetupLRUQueue(int _size) {
		size = _size;
		queue = new T[size];
		/*for (int i = 0; i < size; ++i)
			queue[i] = nullptr;*/
		front = 0;
		rear = size - 1;
	}

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

	~LRUQueue() {
		/*while ((rear + 1) % size == front) {
			Free(queue[front]);
			front = (front + 1) % size;
		}
		delete[] queue;*/
	}
};