#pragma once
#include "RedundancyEliminator_CPP.h"

class RedundancyEliminator_CPP_CollisionTest: RedundancyEliminator_CPP
{
private:

public:
	RedundancyEliminator_CPP_CollisionTest();
	~RedundancyEliminator_CPP_CollisionTest();

	inline unsigned long long computeChunkHash(char* chunk, unsigned int chunkSize);
	unsigned int fingerPrinting(deque<unsigned int> indexQ, char* package) override;
};

