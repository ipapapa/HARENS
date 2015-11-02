#pragma once
#include "Definition.h"

class TreeHash
{
public:
	unsigned long long* keys;
	unsigned int* vals;
	static const size_t POW_OF_2;
	static const size_t MAX_HASH_OFFSET;
	static const unsigned int MOD_NUM;
	static const unsigned int LOCATION_MODS[];

	TreeHash();
	~TreeHash();

	/*Return the location of the key, if not exist, return -1*/
	int Find(unsigned long long key);

	/*Return if adding is succesful*/
	bool InsertNew(unsigned long long key);

	void Reduce(unsigned long long key);
};

