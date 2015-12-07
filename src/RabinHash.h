#pragma once
#include "Definition.h"
using namespace std;

class RabinHash {
private:
    //Tables to store 2 << n where n -66 to 88 computations
    unsigned int *TA,
				 *TB,
				 *TC,
				 *TD;
	//Tables that combines two integers into a long number
	unsigned long long *TALONG,
					   *TBLONG,
					   *TCLONG,
					   *TDLONG;
    //Irreducible polynomial used in the finger printing algorithm
    unsigned long long pt;

	/*
	* Initialize the tables to zero
	*/
	void initialize(unsigned int* T);

	/*
	* Function that fills the polynomial tables
	*/
	void initializePolynomial(unsigned int* T, unsigned long long* TLONG, int shiftBit);

    //Polynomail Generation Functions
	/*
	* According to "Probabilistic algorithms in finite fields" by Michael Rabin,
	* the probability of a random selected degree-n polynoial to be irreducible is 1/n,
	* so the expected number of polynomials we need to pick before finding the correct one is n.
	*/
	unsigned long long genIrreduciblePoly();

	/*
	* Reference: "Tests and Constructions of Irreducible Polynomials over Finite Fileds"
	* by Shuhong Gao and Daniel Panario.
	* The pseudo-code of this function is the following ("^" means "to the power of"):
	***************************************************************************************
	* for j := 1 to k do
	* 	n_j = n / p_j (distinct prime divisor of n)
	* for i := 1 to k do
	* 	g := gcd(polynomial, x^(q^n_i)) - x mod polynomial)
	* 	if g != 1, then f is reducible
	* end for
	* g := x^(q^n) - x mod polynomial
	* if g = 0, then f is irreducible
	* else, f is reducible
	****************************************************************************************
	*/
	bool isIrreducible(unsigned long long polynomial);

	/*
	* This function compute the square of a polynomial and mod a polynomial a specific times.
	* It's done by the following steps:
	*	1. suppose poly = sumof(a_i 2^i), a_i = 0 or 1
	* 	2. so poly*poly mod module = sumof(a_i * shiftleftAndMod(poly, i, module))
	* 	3. repeat 1 and 2 a specific times, because poly_times = poly_(times - 1)^2; poly_(times - 1) = poly_(times - 2)^2; ...
	*/
	unsigned long long squareAndModManyTimes(unsigned long long poly, unsigned long long module, int times);

	/*
	* This function is based on the fact that both polynomials and mod are 63 degree polynomails,
	* represented by 64-bit numbers (the first bit is 1).
	*/
	unsigned long long shiftLeftAndMod(unsigned long long number, int shiftBit, unsigned long long mod);

	/*
	* Compute the greatest common divisor of two polynomials represented by two unsigned long long number
	*/
	unsigned long long gcd(unsigned long long num1, unsigned long long num2);

	/*
	* Count the number of bits from the first 1 to the end
	*/
    inline int bitsCount(unsigned long long num);

public:
	static const unsigned int TABLE_ROW_NUM = 256;
	static const unsigned int TABLE_COL_NUM = 2;
    // intialize the left shift arrays
    RabinHash();
	void print();

	/*
	* The Rabin hash function. argument "strLen" must be a multiple of 4
	*/
	unsigned long long Hash(const char* str, unsigned int strLen);
    ~RabinHash();

	//The following are for CUDA Implementation
	unsigned long long* GetTALONG() const;
	unsigned long long* GetTBLONG() const;
	unsigned long long* GetTCLONG() const;
	unsigned long long* GetTDLONG() const;
};
