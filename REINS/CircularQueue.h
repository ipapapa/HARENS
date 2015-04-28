#pragma once
#include "Definition.h"
using namespace std;

/*
We have 2 threads accessing an object of this class simultaneously
one for push, the other for pop
*/
template <class T>
class CircularQueue
{
public:
	T* queue;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	uint size;
	mutex contentMutex;
	condition_variable contentCond;

	CircularQueue() {}

	CircularQueue(int _size) {
		size = _size;
		queue = new T[size];
		front = 0;
		rear = size - 1;
		//The full situation is front == (rear + 2) % size
	}

	CircularQueue& operator=(CircularQueue obj) {
		this->queue = obj.queue;
		this->front = obj.front;
		this->rear = obj.rear;
		this->size = obj.size;
		return *this;
	}

	void SetupUcharArrayCircularQueue() {
		for (int i = 1; i < size; ++i)
			queue[i] = new uchar[SHA256_DIGEST_LENGTH];
		queue[rear] = (T)0;
	}

	void Push(T hashValue) {
		//Make sure that the queue is not full
		unique_lock<mutex> contentLock(contentMutex);
		if ((rear + 2) % size == front) {
			contentCond.wait(contentLock);
		}
		contentLock.unlock();

		rear = (rear + 1) % size;
		queue[rear] = hashValue;
		//notify pop that one entry is added into queue
		contentCond.notify_one();	
	}

	/*Latent Push only work on the situation that T is a pointer*/
	T LatentPush() {
		unique_lock<mutex> contentLock(contentMutex);
		if ((rear + 2) % size == front) {
			contentCond.wait(contentLock);
		}
		contentLock.unlock();

		rear = (rear + 1) % size;
		//notify pop that one entry is added into queue
		contentCond.notify_one();
		return queue[rear];
	}

	T Pop() {
		//Make sure that the queue is not empty
		unique_lock<mutex> contentLock(contentMutex);
		if ((rear + 1) % size == front) {
			contentCond.wait(contentLock);
		}
		contentLock.unlock();

		T ret = queue[front];
		front = (front + 1) % size;
		contentCond.notify_one();
		return ret;
	}

	~CircularQueue() {
		//delete[] queue;
	}
};

