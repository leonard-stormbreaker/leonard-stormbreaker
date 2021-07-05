#pragma once

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Polynomial {
public:
	Polynomial() : max_power_(0), coeffs_({}) {}
	Polynomial(size_t power);
	Polynomial(const vector<long double>& coeffs) : max_power_(coeffs.size() - 1), coeffs_(coeffs) {
	/*	if(coeffs.size() == 1) {
			max_power_ = 0;
			coeffs_ = coeffs;
			cout << "zdes" << endl;
		} else {
		max_power_ = coeffs.size() - 1;
		coeffs_ = coeffs;
		cout << "ili zdes" << endl;
		}*/
	}
	Polynomial(const Polynomial& other);

	void SetDegree(size_t power);
	size_t GetDegree() const;

	const long double& operator[](const size_t& pos) const;
	long double& operator[](const size_t& pos);

	Polynomial& operator=(const Polynomial& other);
	Polynomial& operator=(Polynomial&& other);

	void SwitchSigns();

	const vector<long double>& GetVector() const;

	void Shrink(); //убрать нулевые коэффициенты
	long double operator()(const long double& x) const;
private:
	size_t max_power_;
	vector<long double> coeffs_;
};

void ShrinkVector(vector<long double>& vec);

Polynomial Differentiate(const Polynomial& poly);

Polynomial operator+(const Polynomial& lhs, const Polynomial& rhs);
Polynomial operator%(const Polynomial& numerator, const Polynomial& denominator);
Polynomial operator*(const long double& coef, const Polynomial& rhs);
Polynomial operator*(const Polynomial& lhs, const Polynomial& rhs);

istream& operator>>(istream& is, Polynomial& poly);
ostream& operator<<(ostream& os, const Polynomial& poly);
