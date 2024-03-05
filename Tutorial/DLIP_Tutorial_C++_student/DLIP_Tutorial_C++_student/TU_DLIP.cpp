#include "TU_DLIP.h"

#include <iostream>

// =============================
// Exercise 1 :: Define Function
// =============================

int sum(int val1, int val2)
{
	// Add code here

	return(val1 + val2);

}

int MyNum::sum(int val1, int val2) {

	return(val1 + val2);

}

MyNum::MyNum(int x) {
	num = x;
}
MyNum::MyNum(int x, int y) {
	num = x;
	C = y;
}

void MyNum::print() {
	std::cout << "print" << std::endl;
}
