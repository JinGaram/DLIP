#include <iostream>
#include"TU_DLIP.h"

namespace proj_A
{
	class myNum {
	public:
		int val1;
		int val2;
		int val3;
		
		myNum(int in1, int in2, int in3) {
			val1 = in1;
			val2 = in2;
			val3 = in3;


		}
	};
	
}

namespace proj_B
{
	// Add code here

}


void main()
{
	proj_A::myNum mynum1(1, 2, 3);
	proj_B::myNum mynum2(4, 5, 6);

	mynum1.print();
	mynum2.print();

	system("pause");
}