#pragma once
#include <unordered_map>
#include <openssl/sha.h>
#include "Definition.h"

class VirtualHash
{
protected:
	struct my_equal_to {
		bool operator()(const uchar* __x, const uchar* __y) const {
			for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
				if (__x[i] != __y[i])
					return false;
			}
			return true;
		}
	};

	struct Hash_Func {
		//BKDR hash algorithm
		int operator()(uchar * str)const {
			int seed = 131;//31  131 1313 13131131313 etc//
			int hash = 0;
			for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
				hash = (hash * seed) + str[i];
			}

			return hash & (0x7FFFFFFF);
		}
	};

	struct CircularQueue {
		uchar** queue;
		int front, rear;	//rear point to the last used entry
		uint size;

		CircularQueue() {}

		CircularQueue(int _size) {
			size = _size;
			queue = new uchar*[size];
			front = 0;
			rear = 0;
			queue[rear] = NULL;
		}
		
		void SetupCircularQueue(int _size) {
			size = _size;
			queue = new uchar*[size];
			front = 0;
			rear = 0;
			queue[rear] = NULL;
		}

		uchar* Add(uchar* hashValue) {
			uchar* to_be_del = NULL;
			if ((rear + 1) % size == front) {
				to_be_del = queue[front];
				front = (front + 1) % size;
			}
			rear = (rear + 1) % size;
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

	typedef std::unordered_map<uchar*, uint, Hash_Func, my_equal_to> charPtMap;
	uint size;

public:
	VirtualHash() {}
	VirtualHash(uint _size);
	void SetupVirtualHash(uint _size);
	~VirtualHash();

	virtual uchar* Add(uchar* hashValue) = 0;

	virtual bool Find(uchar* hashValue) = 0;
};

