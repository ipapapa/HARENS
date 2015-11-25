#pragma once
#include "Definition.h"
using namespace std;

class RabinHash {
private:
    /* Tables to store 2 << n where n -66 to 88 computations */
    unsigned int* TA;
	unsigned int* TB;
	unsigned int* TC;
	unsigned int* TD;
	unsigned long long* TALONG;
	unsigned long long* TBLONG;
	unsigned long long* TCLONG;
	unsigned long long* TDLONG;
    /* Irreducible polynomial used in the finger printing algorithm */
    unsigned long long pt;

	void initialize(unsigned int* T);
	void initializePolynomial(unsigned int* T, unsigned long long* TLONG, int shiftBit);

    /*
     * Polynomail Generation Functions from Kelu
    */
    unsigned long long genIrreduciblePoly();
    bool isIrreducible(unsigned long long polynomial);
    unsigned long long squareAndModManyTimes(unsigned long long poly, unsigned long long module, int times);
    unsigned long long shiftLeftAndMod(unsigned long long number, int shiftBit, unsigned long long mod);
    unsigned long long gcd(unsigned long long num1, unsigned long long num2);
    inline int bitsCount(unsigned long long num);

public:
	static const unsigned int TABLE_ROW_NUM = 256;
	static const unsigned int TABLE_COL_NUM = 2;
    /* intialize the left shift arrays */
    RabinHash();
    void print();
	/* strLen must be a multiple of 4 */
	unsigned long long Hash(const char* str, unsigned int strLen);
    ~RabinHash();
	/*The following are for CUDA Implementation*/
	unsigned long long* GetTALONG() const;
	unsigned long long* GetTBLONG() const;
	unsigned long long* GetTCLONG() const;
	unsigned long long* GetTDLONG() const;
};
