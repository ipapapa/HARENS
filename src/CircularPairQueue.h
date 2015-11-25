#pragma once
#include "Definition.h"
using namespace std;

/*
We have 2 threads accessing an object of this class simultaneously, one for push, the other for pop.
I put a few optimizations in this class. It would not suit any pairs any more. 
*/
template <class S, class T>
class CircularPairQueue
{
public:
	S* firstQ;
	T* secondQ;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	unsigned int size;
	mutex contentMutex;
	mutex fullMutex;
	condition_variable fullCond;

	CircularPairQueue() {
		size = TEST_MAX_KERNEL_INPUT_LEN;
		firstQ = new S[size];
		secondQ = new T[size];
		front = 0;
		rear = size - 1;
		//The full situation is front == (rear + 2) % size
	}

	CircularPairQueue(int _size) {
		size = _size;
		firstQ = new S[size];
		secondQ = new T[size];
		front = 0;
		rear = size - 1;
		//The full situation is front == (rear + 2) % size
	}

	/*CircularUintQueue& operator=(CircularUintQueue obj) {
	this->queue = obj.queue;
	this->front = obj.front;
	this->rear = obj.rear;
	this->size = obj.size;
	return *this;
	}*/

	void Push(S firstHashValue, T secondHashValue) {
		//Make sure that the queue is not full
		unique_lock<mutex> fullLock(fullMutex);
		if (IsFull()) {
			fullCond.wait(fullLock);
		}
		rear = (rear + 1) % size;

		firstQ[rear] = firstHashValue;
		secondQ[rear] = secondHashValue;
	}
	
	/*This function would return (-1, -1) if the queue is empty*/
	void Pop(S& firstHashValue, T& secondHashValue) {
		if (IsEmpty()) {
			firstHashValue = -1;
			secondHashValue = -1;
			return;
		}

		firstHashValue = firstQ[front];
		secondHashValue = secondQ[front];
		front = (front + 1) % size;
		fullCond.notify_one();
		return;
	}

	inline bool IsEmpty() {
		bool isEmpty;
		unique_lock<mutex> contentLock(contentMutex);
		isEmpty = ((rear + 1) % size == front);
		return isEmpty;
	}

	/*We set the condition of full as ((rear + 2) % size == front)*/
	inline bool IsFull() {
		bool isFull;
		unique_lock<mutex> contentLock(contentMutex);
		isFull = ((rear + 2) % size == front);
		return isFull;
	}

	~CircularPairQueue() {
		delete[] firstQ;
		delete[] secondQ;
	}
};

