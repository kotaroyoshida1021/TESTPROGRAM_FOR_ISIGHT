#include "pch.h"
#include <iostream>
#include <Eigen/Core>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include "Ritz.h"

using namespace std;
using namespace Eigen;


RitzMethod::RitzMethod(int nvar,dbl Length) {
	cout << "RitzInitialize...";
	a = VectorXd::Zero(nvar);
	length = Length;
	dbl q = M_PI / length;
	dbl ds = length / (dbl)(BMAX - 1);
	
	for (int i = 0; i < BMAX; i++) BaseFunctions.push_back(VectorXd::Zero(nvar));
	for (int i = 0; i < BMAX; i++) BaseFunctionsdot.push_back(VectorXd::Zero(nvar));

	for (int i = 0; i < BMAX; i++) {
		VectorXd Base = VectorXd::Zero(nvar);
		VectorXd Basedot = VectorXd::Zero(nvar);
		for (int j = 0; j < nvar; j++) {
			Base(j) = base[j](q, i* ds);
			Basedot(j) = basedot[j](q, i* ds);
		}
		BaseFunctions[i] = Base;
		BaseFunctionsdot[i] = Basedot;
	}
	cout << "done";
}

dbl RitzMethod::Function(dbl s) {
	dbl p = s * (BMAX - 1) / length;
	int n = (int)(p);
	dbl q = p - n;
	if (q == 0.0) {
		return a.dot(BaseFunctions[n]);
	}
	else {
		return (1.0 - q)*a.dot(BaseFunctions[n]) + q * a.dot(BaseFunctions[n + 1]);
	}
}

dbl RitzMethod::operator()(dbl s) {
	dbl p = s * (BMAX - 1) / length;
	int n = (int)(p);
	dbl q = p - n;
	if (q == 0.0) {
		return a.dot(BaseFunctions[n]);
	}
	else {
		return (1.0 - q) * a.dot(BaseFunctions[n]) + q * a.dot(BaseFunctions[n + 1]);
	}
}
//return df/da_i = e_i
dbl RitzMethod::Partial(int i, dbl s) {
	if (i < a.size()) {
		cout << "Error in func " << __func__ << "\n";
		exit(1);
	}
	else {
		dbl p = s * (BMAX - 1) / length;
		int n = (int)(p);
		dbl q = p - n;
		if (q == 0.0) {
			return BaseFunctions[n](i);
		}
		else {
			return (1.0 - q) * BaseFunctions[n](i) + q * BaseFunctions[n + 1](i);
		}
	}
}

void RitzMethod::terminates() {
	vector<VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>>().swap(BaseFunctions);
}
//return df/ds = sum_i^n a_i de_i/ds
dbl RitzMethod::Derivative(dbl s) {
	dbl p = s * (BMAX - 1) / length;
	int n = (int)(p);
	dbl q = p - n;
	if (q == 0.0) {
		return a.dot(BaseFunctionsdot[n]);
	}
	else {
		return (1.0 - q) * a.dot(BaseFunctionsdot[n]) + q * a.dot(BaseFunctionsdot[n + 1]);
	}
}