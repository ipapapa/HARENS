#include "RabinHash.h"


/*
 * Initialize the tables to zero
*/
void RabinHash::initialize(unsigned int* T) {
    for(int i = 0; i < TABLE_ROW_NUM; i++) {
        for(int j = 0; j < TABLE_COL_NUM; j++) {
            T[i * TABLE_COL_NUM + j] = 0;
        }
    }
}

/*
 * Function that fills the polynomial tables
*/
void RabinHash::initializePolynomial(unsigned int* T, unsigned long long* TLONG, int shiftbit) {
	for (int i = 0; i < TABLE_ROW_NUM; i++) {
        unsigned long long currValue = shiftLeftAndMod(i, shiftbit, pt );
		TLONG[i] = currValue;
        T[i * TABLE_COL_NUM + 0] = currValue >> 32;
        T[i * TABLE_COL_NUM + 1] = (unsigned int)currValue;
    }
}

/*
 * constructor initializes all tables to zero
*/
RabinHash::RabinHash() {
	TA = new unsigned int[TABLE_ROW_NUM * TABLE_COL_NUM];
	TB = new unsigned int[TABLE_ROW_NUM * TABLE_COL_NUM];
	TC = new unsigned int[TABLE_ROW_NUM * TABLE_COL_NUM];
	TD = new unsigned int[TABLE_ROW_NUM * TABLE_COL_NUM];
	TALONG = new unsigned long long[TABLE_ROW_NUM];
	TBLONG = new unsigned long long[TABLE_ROW_NUM];
	TCLONG = new unsigned long long[TABLE_ROW_NUM];
	TDLONG = new unsigned long long[TABLE_ROW_NUM];
    /*initialize(TA);
    initialize(TB);
    initialize(TC);
    initialize(TD);*/
	//This value should have been randomly generated, but we use a fixed value for testing
    pt = 9271613880639673933;			
    initializePolynomial(TA, TALONG, 88);
    initializePolynomial(TB, TBLONG, 80);
    initializePolynomial(TC, TCLONG, 72);
    initializePolynomial(TD, TDLONG, 64);
    //print();
}

void RabinHash::print() {
    for(int i = 0; i < 256; i++) {
        for(int j = 0; j < 2; j++) {
			printf("Values : %d\t%d\t%d\t%d\n", TA[i * TABLE_COL_NUM + j]
				, TB[i * TABLE_COL_NUM + j]
				, TC[i * TABLE_COL_NUM + j] 
				, TD[i * TABLE_COL_NUM + j]);
        }
    }
}

/* strLen must be a multiple of 4 */
unsigned long long RabinHash::Hash(const char* str, unsigned int strLen) {
	/*unsigned long long rabinHash = str[4];
	for (int i = 5; i < 12; ++i)
		rabinHash = (rabinHash << 8) | str[i];
	
	int h, i, j, k;
	h = str[0];
	i = str[1];
	j = str[2];
	k = str[3];

	rabinHash ^= TALONG[h] ^ TBLONG[i] ^ TCLONG[j] ^ TDLONG[k];
	return rabinHash;*/
	unsigned int iter = 0;
	unsigned int w1 = (str[iter] << 24) | (str[iter + 1] << 16) | (str[iter + 2] << 8) | (str[iter + 3]);
	iter += 4;
	unsigned int w2 = (str[iter] << 24) | (str[iter + 1] << 16) | (str[iter + 2] << 8) | (str[iter + 3]);
	iter += 4;
	unsigned int A, h, i, j, k;
	while (iter < strLen) {
		A = 0;
		for (int offset = 0; offset < 4; ++offset) {
			A <<= 8;
			if (iter + offset < strLen)
				A |= str[iter + offset];
		}

        h = (w1 >> 24) & 0xFF;
        i = (w1 >> 16) & 0xFF;
        j = (w1 >> 8) & 0xFF;
        k = w1 & 0xFF;
		w1 = w2 ^ TA[h * TABLE_COL_NUM] ^ TB[i * TABLE_COL_NUM] 
			^ TC[j * TABLE_COL_NUM] ^ TD[k * TABLE_COL_NUM];
		w2 = A ^ TA[h * TABLE_COL_NUM + 1] ^ TB[i * TABLE_COL_NUM + 1]
			^ TC[j * TABLE_COL_NUM + 1] ^ TD[k * TABLE_COL_NUM + 1];
		iter += 4;
    }
    unsigned long long rabinHash = 0;
    rabinHash = (rabinHash << 32) | w1;
    rabinHash = (rabinHash << 32) | w2;
    return rabinHash;
}

/*
 * Destructor
*/
RabinHash::~RabinHash() {
}

/*
 * polynomial Generation function definitions from Kelu
*/
/*
According to "Probabilistic algorithms in finite fields" by Michael Rabin,
the probability of a random selected degree-n polynoial to be irreducible is 1/n,
so the expected number of polynomials we need to pick before finding the correct one is n.
*/
unsigned long long RabinHash::genIrreduciblePoly() {
	srand ((unsigned int)time(NULL));
    long long randomNum = 0;
	do {
	    //Generate a random polynomial with 63 degree
        randomNum = (rand() % (1 << 7)) + (1 << 7);	//the first bit have to be 1
		while(randomNum > 0)
			randomNum = (randomNum << 8) + (rand() % (1 << 8));
	} while(!isIrreducible(randomNum));
	return randomNum;
}

/*
Reference: "Tests and Constructions of Irreducible Polynomials over Finite Fileds" by Shuhong Gao and Daniel Panario.
The pseudo-code of this function is the following ("^" means "to the power of"):
for j := 1 to k do
	n_j = n / p_j (distinct prime divisor of n)
for i := 1 to k do
	g := gcd(polynomial, x^(q^n_i)) - x mod polynomial)
	if g != 1, then f is reducible
end for
g := x^(q^n) - x mod polynomial
if g = 0, then f is irreducible
else, f is reducible
*/
bool RabinHash::isIrreducible(unsigned long long polynomial) {
	//x^2 - x (mod polynomial)
	unsigned long long rightNum = 2;                               //set rightNum = x
	rightNum = squareAndModManyTimes(rightNum, polynomial, 1);    //x square
	rightNum ^= 2;                                                 //set rightNum = x^2 - x
	unsigned long long g = gcd(polynomial, rightNum);
	if(g != 1)
		return false;
	//Compute g = x^(2^63) - x mod polynomial
	//set g_63 = x^(2^63) mod polynomial
	unsigned long long g_63 = squareAndModManyTimes(2, polynomial, 63);

	//set g = x^(2^63) - x mod polynomial
	g = g_63 ^ 2;
	if(g == 0)
		return true;
	else
		return false;
}

/*
This function compute the square of a polynomial and mod a polynomial a specific times
it's done by the following steps:
    1. suppose poly = sumof(a_i 2^i), a_i = 0 or 1
	2. so poly*poly mod module = sumof(a_i * shiftleftAndMod(poly, i, module))
	3. repeat 1 and 2 a specific times, because poly_times = poly_(times - 1)^2; poly_(times - 1) = poly_(times - 2)^2; ...
*/
unsigned long long RabinHash::squareAndModManyTimes(unsigned long long poly, unsigned long long module, int times) {
    unsigned long long poly_plusonetime = poly;
    for(; times > 0; --times) {
        poly = poly_plusonetime;
        poly_plusonetime = 0;
        //set poly_plusonetime = poly^2 mod module
        for(int i = 0; i < 64; ++i) {
            if( ((poly >> i) & 1) != 0) //the i'th bit of poly
                poly_plusonetime ^= shiftLeftAndMod(poly, i, module);
        }
    }
    return poly_plusonetime;
}

//This is based on the fact that both polynomials and mod are 63 degree polynomails, represented by 64-bit numbers (the first bit is 1).
unsigned long long RabinHash::shiftLeftAndMod(unsigned long long number, int shiftBit, unsigned long long mod) {
	do {
		while((number >> 63) != 1 && shiftBit > 0) {
			number <<= 1;
			--shiftBit;
		}
		//number % mod should be less than 63-degree polynomial, which means the first bit should be 0
		if((number >> 63) == 1)
            number ^= mod;
	} while(shiftBit > 0);
	return number;
}

/*
Compute the greatest common divisor of two polynomials represented by two unsigned long long number
*/
unsigned long long RabinHash::gcd(unsigned long long num1, unsigned long long num2) {
	unsigned long long nums[2];
	if(num1 > num2) {
		nums[0] = num1;
		nums[1] = num2;
	}
	else {
		nums[0] = num2;
		nums[1] = num1;
	}
	int i = 0;
	while(nums[1 - i] != 0) {
		//set nums[i] = nums[i] % nums[1 - i];
		while(bitsCount(nums[i]) >= bitsCount(nums[1 - i])) {
			nums[i] ^= nums[1 - i] << (bitsCount(nums[i]) - bitsCount(nums[1 - i]));
		}
		i = 1 - i;
	}
	return nums[i];
}

/*
Count the number of bits from the first 1 to the end
*/
inline int RabinHash::bitsCount(unsigned long long num) {
	int count = 0;
	while(num != 0) {
		num >>= 1;
		++count;
	}
	return count;
}

/*The following are for CUDA Implementation*/
unsigned long long* RabinHash::GetTALONG() const {
	return TALONG;
}
unsigned long long* RabinHash::GetTBLONG() const {
	return TBLONG;
}
unsigned long long* RabinHash::GetTCLONG() const {
	return TCLONG;
}
unsigned long long* RabinHash::GetTDLONG() const {
	return TDLONG;
}