#include "RabinHash.h"


/*
 * Initialize the tables to zero
*/
void RabinHash::initialize(uint* T) {
    for(int i = 0; i < TABLE_ROW_NUM; i++) {
        for(int j = 0; j < TABLE_COL_NUM; j++) {
            T[i * TABLE_COL_NUM + j] = 0;
        }
    }
}

/*
 * Function that fills the polynomial tables
*/
void RabinHash::initializePolynomial(uint* T, ulong* TLONG, int shiftbit) {
	for (int i = 0; i < TABLE_ROW_NUM; i++) {
        ulong currValue = shiftLeftAndMod(i, shiftbit, pt );
		TLONG[i] = currValue;
        T[i * TABLE_COL_NUM + 0] = currValue >> 32;
        T[i * TABLE_COL_NUM + 1] = (uint)currValue;
    }
}

/*
 * constructor initializes all tables to zero
*/
RabinHash::RabinHash() {
	TA = new uint[TABLE_ROW_NUM * TABLE_COL_NUM];
	TB = new uint[TABLE_ROW_NUM * TABLE_COL_NUM];
	TC = new uint[TABLE_ROW_NUM * TABLE_COL_NUM];
	TD = new uint[TABLE_ROW_NUM * TABLE_COL_NUM];
	TALONG = new ulong[TABLE_ROW_NUM];
	TBLONG = new ulong[TABLE_ROW_NUM];
	TCLONG = new ulong[TABLE_ROW_NUM];
	TDLONG = new ulong[TABLE_ROW_NUM];
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
			cout << "Values : " << TA[i * TABLE_COL_NUM + j] << "\t" 
				<< TB[i * TABLE_COL_NUM + j] << "\t" 
				<< TC[i * TABLE_COL_NUM + j] << "\t" 
				<< TD[i * TABLE_COL_NUM + j] << endl;
        }
    }
}

/* strLen must be a multiple of 4 */
ulong RabinHash::Hash(const char* str, uint strLen) {
	/*ulong rabinHash = str[4];
	for (int i = 5; i < 12; ++i)
		rabinHash = (rabinHash << 8) | str[i];
	
	int h, i, j, k;
	h = str[0];
	i = str[1];
	j = str[2];
	k = str[3];

	rabinHash ^= TALONG[h] ^ TBLONG[i] ^ TCLONG[j] ^ TDLONG[k];
	return rabinHash;*/
	uint iter = 0;
	uint w1 = str[iter++];
	uint w2 = str[iter++];
	uint A, h, i, j, k;
	while (iter < strLen) {
        A = str[iter++];

        h = (w1 >> 24) & 0xFF;
        i = (w1 >> 16) & 0xFF;
        j = (w1 >> 8) & 0xFF;
        k = w1 & 0xFF;
		w1 = w2 ^ TA[h * TABLE_COL_NUM] ^ TB[i * TABLE_COL_NUM] 
			^ TC[j * TABLE_COL_NUM] ^ TD[k * TABLE_COL_NUM];
		w2 = A ^ TA[h * TABLE_COL_NUM + 1] ^ TB[i * TABLE_COL_NUM + 1]
			^ TC[j * TABLE_COL_NUM + 1] ^ TD[k * TABLE_COL_NUM + 1];
    }
    ulong rabinHash = 0;
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
ulong RabinHash::genIrreduciblePoly() {
	srand ((uint)time(NULL));
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
bool RabinHash::isIrreducible(ulong polynomial) {
	//x^2 - x (mod polynomial)
	ulong rightNum = 2;                               //set rightNum = x
	rightNum = squareAndModManyTimes(rightNum, polynomial, 1);    //x square
	rightNum ^= 2;                                                 //set rightNum = x^2 - x
	ulong g = gcd(polynomial, rightNum);
	if(g != 1)
		return false;
	//Compute g = x^(2^63) - x mod polynomial
	//set g_63 = x^(2^63) mod polynomial
	ulong g_63 = squareAndModManyTimes(2, polynomial, 63);

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
ulong RabinHash::squareAndModManyTimes(ulong poly, ulong module, int times) {
    ulong poly_plusonetime = poly;
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
ulong RabinHash::shiftLeftAndMod(ulong number, int shiftBit, ulong mod) {
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
Compute the greatest common divisor of two polynomials represented by two ulong number
*/
ulong RabinHash::gcd(ulong num1, ulong num2) {
	ulong nums[2];
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
inline int RabinHash::bitsCount(ulong num) {
	int count = 0;
	while(num != 0) {
		num >>= 1;
		++count;
	}
	return count;
}

/*The following are for CUDA Implementation*/
ulong* RabinHash::GetTALONG() const {
	return TALONG;
}
ulong* RabinHash::GetTBLONG() const {
	return TBLONG;
}
ulong* RabinHash::GetTCLONG() const {
	return TCLONG;
}
ulong* RabinHash::GetTDLONG() const {
	return TDLONG;
}