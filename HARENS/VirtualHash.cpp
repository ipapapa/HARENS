#include "VirtualHash.h"

template <int str_len>
VirtualHash<str_len>::VirtualHash<str_len>(unsigned int _size)
{
	size = _size;
}

template <int str_len>
void VirtualHash<str_len>::SetupVirtualHash(unsigned int _size)
{
	size = _size;
}

template <int str_len>
VirtualHash<str_len>::~VirtualHash()
{
}
