#pragma once
#include "Definition.h"

class VirtualHash
{
protected:
	struct my_equal_to {
		bool operator()(const uchar* __x, const uchar* __y) const {
			for (int i = 0; i < SHA_DIGEST_LENGTH; ++i) {
				if (__x[i] != __y[i])
					return false;
			}
			return true;
		}
	};

	struct Hash_Func {
		//BKDR hash algorithm
		int operator()(uchar * str)const {
			int seed = 131;/*31  131 1313 13131131313 etc*/
			int hash = 0;
			for (int i = 0; i < SHA_DIGEST_LENGTH; ++i) {
				hash = (hash * seed) + str[i];
			}

			return hash & (0x7FFFFFFF);
		}
	};

	typedef std::unordered_map<uchar*, uint, Hash_Func, my_equal_to> charPtMap;
	uint size;

public:
	VirtualHash() {}
	VirtualHash(uint _size);
	void SetupVirtualHash(uint _size);
	~VirtualHash();

	virtual ulong Add(const ulong hashValue, const bool isDuplicated) = 0;

	virtual bool Find(const ulong hashValue) = 0;

	virtual bool FindAndAdd(const ulong& hashValue, ulong& toBeDel) = 0;
};