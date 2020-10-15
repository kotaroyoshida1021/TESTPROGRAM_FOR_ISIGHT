#pragma once
#include "pch.h"
#include <vector>

typedef enum GLI_NUM {
	GLI_5, GLI_10, GLI_15, GLI_20, GLI_30,
	GLI_40, GLI_50, GLI_60, GLI_100
} GLI_NUM_t;

using namespace std;
using namespace Eigen;

class Coordinates;
template<typename T>
class GaussIntegral {
private:
	GLI_NUM_t G_NUM;// determine the number of division for Gaussintergal
	int NUM;
	vector<dbl> wj; //weights
	vector<dbl> xj; //coefficients
public:
	void SetGaussIntegralParams(GLI_NUM_t NUM);
	void Gauss_ForClass(vector<dbl>& x, vector<dbl>& w); //do not use
	int ShowNum(); //do not use
	T GaussIntegralFunc(dbl init, dbl end, function<T(dbl)> func); //Well used.
	T GaussIntegralFunc2Dmarginal1(dbl x_init, dbl x_end, dbl y, function<dbl(dbl, dbl)> func); // for double integral 
	T GaussIntegralFunc2Dmarginal2(dbl x, dbl y_init, dbl y_end, function<dbl(dbl, dbl)> func); // for double integral 
};
