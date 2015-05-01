#pragma once
#include "Definition.h"
using namespace std;

/*
We have 2 threads accessing an object of this class simultaneously
one for push, the other for pop
*/
class CircularUcharArrayQueue
{
public:
	uchar** queue;
	int front, rear;	//rear point to the last used entry, there's an empty entry after rear
	uint size;
	mutex contentMutex;
	condition_variable contentCond;

	CircularUcharArrayQueue() {
		size = TEST_MAX_KERNEL_INPUT_LEN;
		queue = new uchar*[size];
		for (int i = 0; i < size; ++i)
			queue[i] = new uchar[SHA_DIGEST_LENGTH];
		front = 0;
		rear = size - 1;
		//The full situation is front == (rear + 2) % size
	}

	CircularUcharArrayQueue(int _size) {
		size = _size;
		queue = new uchar*[size];
		for (int i = 0; i < size; ++i)
			queue[i] = new uchar[SHA_DIGEST_LENGTH];
		front = 0;
		rear = size - 1;
		//The full situation is front == (rear + 2) % size
	}

	/*CircularUcharArrayQueue& operator=(CircularUcharArrayQueue obj) {
		this->queue = obj.queue;
		this->front = obj.front;
		this->rear = obj.rear;
		this->size = obj.size;
		return *this;
	}*/

	void Push(uchar* hashValue) {
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

	/*Latent Push only work on the situation that uchar* is a pointer*/
	uchar* LatentPush() {
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

	uchar* Pop() {
		//Make sure that the queue is not empty
		unique_lock<mutex> contentLock(contentMutex);
		if ((rear + 1) % size == front) {
			contentCond.wait(contentLock);
		}
		contentLock.unlock();

		uchar* ret = queue[front];
		front = (front + 1) % size;
		contentCond.notify_one();
		return ret;
	}

	bool IsEmpty() {
		bool isEmpty;
		unique_lock<mutex> contentLock(contentMutex);
		isEmpty = ((rear + 1) % size == front);
		contentLock.unlock();
		contentCond.notify_one();
		return isEmpty;
	}

	~CircularUcharArrayQueue() {
		/*for (int i = 0; i < size; ++i) {
			delete[] queue[i];
		}*/
		delete[] queue;
	}
};

