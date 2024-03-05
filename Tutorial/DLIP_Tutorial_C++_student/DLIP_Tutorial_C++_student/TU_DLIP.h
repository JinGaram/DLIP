#ifndef _TU_DLIP_H		// same as "#if !define _TU_DLIP_H" (or #pragma once) 
#define _TU_DLIP_H

#include <iostream>

// =============================
// Exercise 1 :: Define Function
// =============================

// Add code here
int sum(int val1, int val2);
class MyNum {
	private:
		int A;
		int B;
	public:
		int sum(int val1, int val2);
		int C;
		MyNum(int x);
		MyNum(int x, int y); //Overloading 인자 겹치게 적는것

		void print();

		int num;



};
#endif // !_TU_DLIP_H