#include "TreeHash.h"

const size_t TreeHash::POW_OF_2 = 1048576;
const size_t TreeHash::MAX_HASH_OFFSET = 16;
const unsigned int TreeHash::MOD_NUM = 20;
const unsigned int TreeHash::LOCATION_MODS[] = {
	-1, 0x1, 0x3, 0x7, 0xF,
	0x1F, 0x3F, 0x7F, 0xFF,
	0x1FF, 0x3FF, 0x7FF, 0xFFF,
	0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
	0x1FFFF, 0x3FFFF, 0x7FFFF, 0xFFFFF,
	0x1FFFFF, 0x3FFFFF, 0x7FFFFF, 0xFFFFFF,
	0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF
};

TreeHash::TreeHash()
{	
	keys = new unsigned long long[POW_OF_2 + MAX_HASH_OFFSET];
	vals = new unsigned int[POW_OF_2 + MAX_HASH_OFFSET];
}


TreeHash::~TreeHash()
{
	delete[] keys;
	delete[] vals;
}

/*Return the location of the key, if not exist, return -1*/
int TreeHash::Find(unsigned long long key) {
	int location = key & LOCATION_MODS[MOD_NUM];
	for (int i = 0; i < MAX_HASH_OFFSET; ++i) {
		if (keys[location + i] == key) {
			return location + i;
		}
	}
	return -1;
}

/*Return if adding is succesful*/
bool TreeHash::InsertNew(unsigned long long key) {
	int location = key & LOCATION_MODS[MOD_NUM];
	for (int i = 0; i < MAX_HASH_OFFSET; ++i) {
		if (keys[location + i] == 0) {
			keys[location + i] = key;
			vals[location + i] = 1;
			return true;
		}
	}
	return false;
}

void TreeHash::Reduce(unsigned long long key) {
	int location = key & LOCATION_MODS[MOD_NUM];
	for (int i = 0; i < MAX_HASH_OFFSET; ++i) {
		if (keys[location + i] == key) {
			if ((vals[location + i]--) == 0)
				keys[location + i] = 0;
			return;
		}
	}
}