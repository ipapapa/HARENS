#pragma once
#include "Definition.h"

template <class T>
class CircularQueue
{
public:
	T* queue;
	int front, rear;	//rear point to the last used entry
	uint size;

	CircularQueue() {}

	CircularQueue(int _size) {
		size = _size;
		queue = new T[size];
		front = 0;
		rear = 0;
		queue[rear] = NULL;
	}

	void SetupCircularQueue(int _size) {
		size = _size;
		queue = new T[size];
		front = 0;
		rear = 0;
		queue[rear] = NULL;
	}

	T Add(T hashValue) {
		uchar* to_be_del = NULL;
		if ((rear + 1) % size == front) {
			to_be_del = queue[front];
			front = (front + 1) % size;
		}
		rear = (rear + 1) % size;
		queue[rear] = hashValue;
		return to_be_del;
	}

	~CircularQueue() {
		while (front != rear) {
			delete[] queue[front];
			front = (front + 1) % size;
		}
		delete[] queue;
	}
};

