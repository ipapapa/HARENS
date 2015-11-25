#pragma once
#include "Definition.h"
#include <type_traits>

template<typename T>
class SelfMantainedLRUQueue
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

public:
	T* queue;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	unsigned int size;

	SelfMantainedLRUQueue() {}

	SelfMantainedLRUQueue(int _size) {
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

	~SelfMantainedLRUQueue() {
		/*while ((rear + 1) % size == front) {
			delete[] queue[front];
			front = (front + 1) % size;
		}
		delete[] queue;*/
	}
};