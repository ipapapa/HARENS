#pragma once
#include "Definition.h"

class EncryptionHashes {
public:
	/*
	Compute the hash value of chunk using sha1
	*/
	static void computeSha1Hash(char* chunk, unsigned int chunkSize, unsigned char *hashValue) {
		/*UCHAR* hashValue = new UCHAR[SHA1_HASH_LENGTH];
		SHA((UCHAR*)chunk, chunkSize, hashValue);
		return hashValue;*/
		SHA1((unsigned char*)chunk, chunkSize, hashValue);
	}

	/*
	Compute the hash value of chunk using md5
	*/
	static void computeMd5Hash(char* chunk, unsigned int chunkSize, unsigned char *hashValue) {
		MD5((unsigned char*)chunk, chunkSize, hashValue);
	}
};