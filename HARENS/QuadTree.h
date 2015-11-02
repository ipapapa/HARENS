#pragma once
#include "Definition.h"
#include <iostream>

class QuadTree
{
private:
	class QuadNode {
	public:
		unsigned long long value;
		unsigned int count;
		QuadNode* lowerLevel[4];
		QuadNode* upperLevel;

		QuadNode() {
			value = 0;
			upperLevel = nullptr;
			for (int i = 0; i < 4; ++i)
				lowerLevel[i] = nullptr;
		}

		QuadNode(const unsigned long long _value, const int _count, QuadNode* _upperLevel) {
			value = _value;
			count = _count;
			upperLevel = _upperLevel;
			for (int i = 0; i < 4; ++i)
				lowerLevel[i] = nullptr;
		}

		void Clear() {
			for (int i = 0; i < 4; ++i) {
				if (lowerLevel[i] != nullptr) {
					lowerLevel[i]->Clear();
					delete lowerLevel[i];
				}
			}
		}

		void Split() {
			lowerLevel[value & 0x3] = new QuadNode(value >> 2, count, this);
			value = 0;
			count = 0;
		}

		void AddLowerLevel(const unsigned long long _value) {
			lowerLevel[_value & 0x3] = new QuadNode(_value >> 2, 1, this);
		}

		void Reduce() {
			int validSonNum = 0;
			for (int i = 0; i < 4; ++i) {
				if (lowerLevel[i] == nullptr)
					continue;
				else if (lowerLevel[i]->count < 0)
					delete lowerLevel[i];
				else
					++validSonNum;
			}
			if (validSonNum == 0) {
				count = -1;
				upperLevel->Reduce();
			}
		}

	} *root;

public:
	QuadTree();

	bool FindAndInsert(const unsigned long long hashValue);
	void Erase(const unsigned long long hashValue);

	~QuadTree();

};

