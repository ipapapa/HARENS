#pragma once
#include "Definition.h"
using namespace std;

class RabinHash {
private:
    /* Tables to store 2 << n where n -66 to 88 computations */
    uint* TA;
	uint* TB;
	uint* TC;
	uint* TD;
	ulong* TALONG;
	ulong* TBLONG;
	ulong* TCLONG;
	ulong* TDLONG;
    /* Irreducible polynomial used in the finger printing algorithm */
    ulong pt;

	void initialize(uint* T);
	void initializePolynomial(uint* T, ulong* TLONG, int shiftBit);

    /*
     * Polynomail Generation Functions from Kelu
    */
    ulong genIrreduciblePoly();
    bool isIrreducible(ulong polynomial);
    ulong squareAndModManyTimes(ulong poly, ulong module, int times);
    ulong shiftLeftAndMod(ulong number, int shiftBit, ulong mod);
    ulong gcd(ulong num1, ulong num2);
    inline int bitsCount(ulong num);

public:
	static const uint TABLE_ROW_NUM = 256;
	static const uint TABLE_COL_NUM = 2;
    /* intialize the left shift arrays */
    RabinHash();
    void print();
	/* strLen must be a multiple of 4 */
	ulong Hash(const char* str, uint strLen);
    ~RabinHash();
	/*The following are for CUDA Implementation*/
	ulong* GetTALONG() const;
	ulong* GetTBLONG() const;
	ulong* GetTCLONG() const;
	ulong* GetTDLONG() const;
};
