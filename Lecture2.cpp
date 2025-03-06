#include <iostream>
#include <fstream>
#include "nr3.h"
//your extra include headers

using namespace std;

int main() {
VecDoub xFilip(82); VecDoub yFilip(82);
ifstream Filip("YourSource/FilipData.dat");
for(int i = 0; i < 82; i++) {
	Filip >> yFilip[i];
	Filip >> xFilip[i];
}

VecDoub xPont(40); VecDoub yPont(40);
ifstream Pont("YourSource/PontiusData.dat");
for(int i = 0; i < 40; i++) {
	Pont >> yPont[i];
	Pont >> xPont[i];
}

// your code



return 0;
}
