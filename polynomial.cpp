#include "polynomial.h"

Polynomial::Polynomial(size_t power) {
	max_power_ = power;
	coeffs_.reserve(power + 1);
	coeffs_.assign(power + 1, 0);
}

void Polynomial::SetDegree(size_t power) {
	max_power_ = power;
	coeffs_.resize(power+1);
	coeffs_[0] = 0;
}
size_t Polynomial::GetDegree() const {
	return max_power_;
}

const long double& Polynomial::operator[](const size_t& pos) const {
	return coeffs_[pos];
}
long double& Polynomial::operator[](const size_t& pos) {
	return coeffs_[pos];
}

const vector<long double>& Polynomial::GetVector() const {
	return coeffs_;
}

void Polynomial::Shrink() {
	vector<long double>::reverse_iterator it = coeffs_.rbegin(); //обход с конца
	int counter = 0;
	while(*it == 0) {
		counter++;
		it++;
	}
	coeffs_.resize(max_power_ - counter);
	max_power_ -= counter;
}

long double Polynomial::operator()(const long double& x) const {
	if(this->GetDegree() == 0) {
		return this->coeffs_[0];
	}
	long double res = 0;
	for (auto it = coeffs_.rbegin(); it != coeffs_.rend(); ++it) {
		res *= x;
		res += *it;
	}
	return res;
}

Polynomial Differentiate(const Polynomial& poly) {
	if(poly.GetDegree() == 0) {
		return Polynomial{0};
	}
	Polynomial result{poly.GetDegree() - 1};
	for(size_t i = 0; i < poly.GetDegree(); i++) {
		result[i] = poly[i+1] * (i+1);
	}
	return result;
}

Polynomial operator+(const Polynomial& lhs, const Polynomial& rhs) {
	size_t degree = max(lhs.GetDegree(), rhs.GetDegree());
	size_t border = min(lhs.GetDegree(), rhs.GetDegree());
	vector<long double> res_vec;
	for(size_t i = 0; i < degree + 1; i++) {
		if(i < border + 1) {
			res_vec.push_back(lhs[i] + rhs[i]);
		} else {
			lhs.GetDegree() < rhs.GetDegree() ? res_vec.push_back(rhs[i]) : res_vec.push_back(lhs[i]);
		}
	}
	Polynomial result(res_vec);
	return result;
}

void Polynomial::SwitchSigns() {
	for(size_t i = 0; i < this->coeffs_.size(); i++) {
		this->coeffs_[i] *= -1;
	}
}

Polynomial& Polynomial::operator=(const Polynomial& other) {
	this->SetDegree(other.GetDegree());
	for(size_t i = 0; i < other.GetDegree() + 1; i++) {
		(*this)[i] = other[i];
	}
	return *this;
}
Polynomial& Polynomial::operator=(Polynomial&& other) {
	size_t n = other.GetDegree();
	this->SetDegree(n);
	for(size_t i = 0; i < n + 1; i++) {
		(*this)[i] = move(other[i]);
	}
	return *this;
}
Polynomial::Polynomial(const Polynomial& other) {
	this->SetDegree(other.GetDegree());
	for(size_t i = 0; i < other.GetDegree() + 1; i++) {
		(*this)[i] = other[i];
	}
}

void ShrinkVector(vector<long double>& vec) {
	auto it = vec.rbegin();
	size_t size = vec.size();
	int counter = 0;
	while(*it == 0) {
		counter++;
		it++;
	}
	vec.resize(size - counter);
}

Polynomial operator%(const Polynomial& numerator_, const Polynomial& denominator_) {
	vector<long double> numerator = numerator_.GetVector();
	vector<long double> denominator = denominator_.GetVector();
	size_t degree_num = numerator.size() - 1;
	size_t degree_den = denominator.size() - 1;
	vector<long double> difference = {};
	if(degree_num >= degree_den) {
		while(degree_num >= degree_den) {
			difference.clear();

			auto it_num = numerator.rbegin();
			auto it_den = denominator.rbegin();
			long double multiplier = *it_num / *it_den;
			long double min_ = min(degree_num, degree_den);

			for(size_t i = 0; i < degree_num + 1; i++) {
				i < min_ + 1 ? difference.insert(difference.begin(), *it_num - (multiplier*(*it_den))) :
						difference.insert(difference.begin(), *it_num);
				it_num++;
				it_den++;
			}
			ShrinkVector(difference);
			numerator = difference;
			degree_num = difference.size() - 1;
		}
	} else {
		throw invalid_argument("undividable\n");
	}
	return Polynomial{difference};
}

Polynomial operator*(const long double& coef, const Polynomial& rhs) {
	Polynomial result = rhs;
	for(size_t i = 0; i < rhs.GetDegree() + 1; i++) {
		result[i] *= coef;
	}
	return result;
}

Polynomial operator*(const Polynomial& lhs, const Polynomial& rhs) {
	Polynomial result(lhs.GetDegree() + rhs.GetDegree());

	double b;
	for(size_t k = 0; k < lhs.GetDegree() + 1; k++) {
		b = lhs[k];
		for(size_t i = 0; i < rhs.GetDegree() + 1; i++) {
			result[k+i] += b*rhs[i];
		}
	}
	//result.Shrink();
	return result;
}

istream& operator>>(istream& is, Polynomial& poly) {
	int power;
	is >> power;
	if(power < 0) {
		throw invalid_argument("degree < 0");
	}
	poly.SetDegree(power);
	long double coefficient;
	for(int i = 0; i < power + 1; i++) {
		is >> coefficient;
		poly[i] = coefficient;
	}
	return is;
}

ostream& operator<<(ostream& os, const Polynomial& poly) {
	bool first = true;
	for(size_t i = 0; i < poly.GetDegree() + 1; i++) {
		if(!first) {
			os << " + ";
		}
		first = false;
		if(i == 0) {
			os << "(" << poly[i] << ")";
		} else {
			os << "(" << poly[i] << ")*x^" << i;
		}
	}
	return os;
}
//--------------------------------------------------------------------------------------------------
