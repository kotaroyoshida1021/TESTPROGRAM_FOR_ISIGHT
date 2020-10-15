#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Eigen;
struct ScalarFunction {
public:
	vector<dbl> VEC;
	dbl Ds;
public:
	void set_Info(dbl delS, vector<dbl> &v) {
		Ds = delS;
		VEC = v;
	}
	dbl operator()(dbl s) {
		dbl p = s / Ds;
		int n = int(p);
		dbl q = p - n;
		if (VEC.size() < n) {
			std::cout << "error in func: " << __func__ << "and s = " << s << "\n";
			exit(1);
		}
		if (q == 0.0) return VEC[n];
		else {
			return q * VEC[n + 1] + (1.0 - q)*VEC[n];
		}
	}
	dbl integrand(dbl s) {
		dbl p = s / Ds;
		int n = int(p);
		dbl q = p - n;
		if (VEC.size() < n) {
			std::cout << "error in func: " << __func__ << "and s = " << s << "\n";
			exit(1);
		}

		if (q == 0.0) return VEC[n];
		else {
			return q * VEC[n + 1] + (1.0 - q)*VEC[n];
		}
	}

	dbl derivative(dbl s) {
		dbl sb, sa, h, delS;
		delS = 0.01 * Ds;
		dbl len = Ds * VEC.size();
		if (s + delS > len) {
			sb = s;
			sa = s - delS;
			h = delS;
		}
		else if (s - delS < 0.0) {
			sb = s + delS;
			sa = s;
			h = delS;
		}
		else if (0 <= s - delS && s + delS <= len) {
			sb = s + delS;
			sa = s - delS;
			h = 2.0 * delS;
		}
		else {
			fprintf(stderr, "Error in func :%s", __func__);
			exit(EXIT_FAILURE);
		}
		return (integrand(sb) - integrand(sa)) / h;
	}
};

struct VectorFunction {
public:
	vector<Vector3d, aligned_allocator<Vector3d>> VEC;
	dbl Ds;
public:
	void set_Info(dbl delS, vector<Vector3d, aligned_allocator<Vector3d>> &v) {
		Ds = delS;
		VEC = v;
	}
	Vector3d operator()(dbl s) {
		dbl p = s / Ds;
		int n = int(p);
		dbl q = p - n;
		if (VEC.size() < n) {
			std::cout << "error in func: " << __func__ << "and s = " << s << "\n";
			exit(1);
		}
		if (q == 0.0) return VEC[n];
		else {
			return q * VEC[n + 1] + (1.0 - q)*VEC[n];
		}
	}

	Vector3d integrand(dbl s) {
		dbl p = s / Ds;
		int n = int(p);
		dbl q = p - n;
		if (VEC.size() < n) {
			std::cout << "error in func: " << __func__ << "and s = " << s << "\n";
			exit(1);
		}
		if (q == 0.0) return VEC[n];
		else {
			return q * VEC[n + 1] + (1.0 - q)*VEC[n];
		}
	}
};
template <typename T>
struct VecFunc {
	vector<T, aligned_allocator<T>> VEC;
	dbl Ds;

	void set_Info(dbl dels, vector<T, aligned_allocator<T>> &v) {
		Ds = dels;
		VEC = v;
	}
	T operator()(dbl s) {
		dbl p = s / Ds;
		int n = int(p);
		dbl q = p - n;
		if (VEC.size() < n) {
			std::cout << "error in func: " << __func__ << "and s = " << s << "\n";
			exit(1);
		}
		if (q == 0.0) return VEC[n];
		else {
			return q * VEC[n + 1] + (1.0 - q)*VEC[n];
		}
	}

	T integrand(dbl s) {
		dbl p = s / Ds;
		int n = int(p);
		dbl q = p - n;
		if (VEC.size() < n) {
			std::cout << "error in func: " << __func__ << "and s = " << s << "\n";
			exit(1);
		}
		if (q == 0.0) return VEC[n];
		else {
			return q * VEC[n + 1] + (1.0 - q)*VEC[n];
		}
	}
};