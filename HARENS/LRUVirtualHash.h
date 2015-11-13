#pragma once
#include "Definition.h"

template <int str_len>
struct CharArrayEqualTo {
	bool operator()(const unsigned char* __x, const unsigned char* __y) const {
		for (int i = 0; i < str_len; ++i) {
			if (__x[i] != __y[i])
				return false;
		}
		return true;
	}
};

template <int str_len>
struct CharArrayHashFunc {
	//BKDR hash algorithm
	int operator()(unsigned char * str)const {
		int seed = 131;/*31  131 1313 13131131313 etc*/
		int hash = 0;
		for (int i = 0; i < str_len; ++i) {
			hash = (hash * seed) + str[i];
		}

		return hash & (0x7FFFFFFF);
	}
};

template <int str_len>
class LRUVirtualHash
{
protected:
	typedef std::unordered_map<unsigned char*, 
								unsigned int, 
								CharArrayHashFunc<str_len>, 
								CharArrayEqualTo<str_len>> 
								charPtMap;
	unsigned int size;

public:
	LRUVirtualHash() {}
	LRUVirtualHash(unsigned int _size) { size = _size; }
	void SetupLRUVirtualHash(unsigned int _size) { size = _size; }
	~LRUVirtualHash() {}

	virtual unsigned char* Add(unsigned char* hashValue, const bool isDuplicated) = 0;

	virtual bool Find(unsigned char* hashValue) = 0;

	virtual bool FindAndAdd(unsigned char* hashValue, unsigned char* toBeDel) = 0;
};