#include "profile.h"

#include "splain_maker.h"

#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>
using namespace std;

//vector<long double> xvec{0.17453, 0.52360, 0.87267, 1.22173, 1.57080, 1.91986, 2.26893};
//vector<long double> yvec{12.0/pow(10, 6), 0.00026, 0.00250, 0.01815, 0.09763, 0.40593, 1.38035};

//vector<long double> testx{0, 1, 2, 3, 4, 5, 6};
//vector<long double> testy{1,5,33,109,257,501,865};


int main() {
	ifstream input("variant14_data.txt");
	vector<long double> xvec, yvec;
	string tempS;
	for(int k = 0; k < 2; k++) {
		getline(input, tempS);
		istringstream line(tempS);
		long double temp;
		while(!line.eof()) {
			line >> temp;
			k == 0 ? xvec.push_back(move(temp)) : yvec.push_back(move(temp));
		}
	}
	map<long double, Polynomial> splain_map;
	Polynomial IPoly = Interpoliant(xvec, yvec, xvec.size());

	{   LOG_DURATION("SplainMaker")
		cout << "Интерполяционный многочлен: ";
		cout << fixed << setprecision(3) << IPoly << endl;

		size_t precision = 3;
		cout << "Количество знаков после запятой - " << precision << endl;

		for(size_t i = 0; i < xvec.size() - 1; i++) {
			cout << "[" << xvec[i] << ", " << xvec[i+1] << "]"  << ": ";
			cout << fixed << setprecision(precision) << SplainMaker(xvec[i], xvec[i+1], IPoly) << endl;
			splain_map[xvec[i]] = SplainMaker(xvec[i], xvec[i+1], IPoly);
		}
	}
	while(cin) {
		cout << "Введите значение x: ";
		long double x;
		cin >> x;
		auto it = splain_map.lower_bound(x);
		if(x <= *xvec.rbegin()) {
			cout << "Сплайн: " << setw(5)<< prev(it)->second(x) << endl;
//			cout << "Интерполяционный полином: " << IPoly(x) << endl;
		} else {
			cout << "Сплайн для данного промежутка не найден" << endl;
		}
	}

	return 0;
}
