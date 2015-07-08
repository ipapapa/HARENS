#include "QuadTree.h"


QuadTree::QuadTree()
{
	root = new QuadNode();
}


QuadTree::~QuadTree()
{
	root->Clear();
	delete root;
}

bool QuadTree::FindAndInsert(const unsigned long long hashValue) {
	QuadNode* curNode = root;
	unsigned long long curHash = hashValue;
	while (true) {
		if (curHash == curNode->value) {
			curNode->count++;
			return true;
		}
		else if (curNode->lowerLevel[curHash & 0x3] == nullptr) {
			curNode->Split();
			curNode->AddLowerLevel(curHash);
			return false;
		}
		else {
			curNode = curNode->lowerLevel[curHash & 0x3];	//Take the last 2 digits as the index of subtree
			curHash >>= 2;
		}
	}
}

void QuadTree::Erase(const unsigned long long hashValue) {
	QuadNode* curNode = root;
	unsigned long long curHash = hashValue;
	while (true) {
		if ((curHash & 0x3) == curNode->value) {
			if ((--curNode->count) == 0) {
				curNode->count = -1;
				curNode->upperLevel->Reduce();
			}
			return;
		}
		else {
			curNode = curNode->lowerLevel[curHash & 0x3];	//Take the last 2 digits as the index of subtree
			curHash >>= 2;
		}
	}
}