#pragma once
#include "Definition.h"

class VirtualHash
{
protected:
	struct my_equal_to {
		bool operator()(const unsigned char* __x, const unsigned char* __y) const {
			for (int i = 0; i < SHA_DIGEST_LENGTH; ++i) {
				if (__x[i] != __y[i])
					return false;
			}
			return true;
		}
	};

	struct Hash_Func {
		//BKDR hash algorithm
		int operator()(unsigned char * str)const {
			int seed = 131;/*31  131 1313 13131131313 etc*/
			int hash = 0;
			for (int i = 0; i < SHA_DIGEST_LENGTH; ++i) {
				hash = (hash * seed) + str[i];
			}

			return hash & (0x7FFFFFFF);
		}
	};

	typedef std::unordered_map<unsigned char*, unsigned int, Hash_Func, my_equal_to> charPtMap;
	unsigned int size;

public:
	VirtualHash() {}
	VirtualHash(unsigned int _size);
	void SetupVirtualHash(unsigned int _size);
	~VirtualHash();

	virtual unsigned char* Add(unsigned char* hashValue, const bool isDuplicated) = 0;

	virtual bool Find(unsigned char* hashValue) = 0;

	virtual bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel) = 0;
};