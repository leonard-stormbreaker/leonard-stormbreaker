#pragma once

#include "polynomial.h"

using namespace std;

Polynomial Interpoliant(vector<long double> xvec_, vector<long double> yvec_, size_t power_splain) {
	size_t n = xvec_.size();

	vector<long double> Fx = yvec_;
	vector<long double> newFx;

	vector<long double> d; d.push_back(1);
	Polynomial D(d);
	vector<long double> p; p.push_back(yvec_[0]);
	Polynomial P(p);


	for(size_t j = 1; j < n; j++) {
		for(size_t i = 0; i < n - j; i++) {
			newFx.push_back((Fx[i+1] - Fx[i])/(xvec_[i+j] - xvec_[i]));
		}
		Polynomial temp({(-1)*xvec_[j-1], 1});
		D = temp * D;
		P = P + newFx[0]*D;
		Fx = newFx;
		newFx.clear();
		if(j == power_splain) {
			break;
		}
	}
	P.Shrink();
	return P;
}

Polynomial SplainMaker(long double x1_, long double x2_, Polynomial P) {
	vector<long double> a_coeffs;
	Polynomial dP = Differentiate(P);
	long double a3 = (dP(x2_) * (x2_ - x1_) - 2*(P(x2_) - P(x1_)) + dP(x1_) * (x2_ - x1_))
			/pow(x2_ - x1_, 3);

	long double a2 = (((-1)*dP(x2_)*(x2_ - x1_)*(x2_ + 2*x1_) + 3*(P(x2_) - P(x1_))*(x2_+ x1_))
			/(pow(x2_ - x1_, 3))) - ((dP(x1_)*(x2_-x1_)*(x1_ + 2*x2_))/(pow(x2_-x1_,3)));

	long double a1 = ((dP(x2_)*x1_*(2*x2_+x1_)*(x2_-x1_)-6*(P(x2_)-P(x1_))*x1_*x2_)
			/(pow(x2_-x1_,3))) + ((dP(x1_)*x2_*(x2_+2*x1_)*(x2_-x1_))/(pow(x2_-x1_,3)));

	long double a0 = (((-1)*dP(x2_)*x1_*x1_*x2_*(x2_-x1_) + P(x2_)*x1_*x1_*(3*x2_-x1_))/(pow(x2_-x1_,3)))
			+((P(x1_)*x2_*x2_*(x2_-3*x1_) - dP(x1_)*x1_*x2_*x2_*(x2_-x1_))/(pow(x2_-x1_,3)));
	a_coeffs.push_back(move(a0));
	a_coeffs.push_back(move(a1));
	a_coeffs.push_back(move(a2));
	a_coeffs.push_back(move(a3));

	return Polynomial{move(a_coeffs)};

}
