#include "pch.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <functional>
#include "GaussIntegral.h"
#include "G_L_Params.h"

using namespace std;
using namespace Eigen;

int convertGLInum_integer(GLI_NUM_t GLInum) {
	switch (GLInum) {
	case GLI_5: return  5;	break;
	case GLI_10: return 10;	break;
	case GLI_15: return 15;	break;
	case GLI_20: return 20;	break;
	case GLI_30: return 30;	break;
	case GLI_40: return 40;	break;
	case GLI_50: return 50;	break;
	case GLI_60: return 60;	break;
	case GLI_100: return 100;	break;
	default:
		fprintf(stderr, "\t error in func.\"%s()\"\n", __func__);
		fprintf(stderr, "\t GLInum = %d \n", GLInum);
		exit(EXIT_FAILURE);
	}
}

template void GaussIntegral<Vector3d>::SetGaussIntegralParams(GLI_NUM_t GLI_NUM);
template void GaussIntegral<Vector3d>::Gauss_ForClass(vector<dbl>& x, vector<dbl>& w);
template int GaussIntegral<Vector3d>::ShowNum();
//template Vector3d GaussIntegral<Vector3d>::GaussIntegralFunc(dbl init, dbl end, Vector3d (*func)(dbl s));
template Vector3d GaussIntegral<Vector3d>::GaussIntegralFunc(dbl init, dbl end, function<Vector3d(dbl)> func);


template void GaussIntegral<dbl>::SetGaussIntegralParams(GLI_NUM_t GLI_NUM);
template dbl GaussIntegral<dbl>::GaussIntegralFunc(dbl init, dbl end, function<dbl(dbl)> func);

template void GaussIntegral<Vector2d>::SetGaussIntegralParams(GLI_NUM_t GLI_NUM);
template Vector2d GaussIntegral<Vector2d>::GaussIntegralFunc(dbl init, dbl end, function<Vector2d(dbl)> func);
template<typename T>
void GaussIntegral<T>::SetGaussIntegralParams(GLI_NUM_t GLI_NUM) {
	G_NUM = GLI_NUM;
	NUM = convertGLInum_integer(G_NUM);
	switch (G_NUM) {
	case GLI_5: xj = xi5; wj = w5; break;
	case GLI_10:  xj = xi10; wj = w10; break;
	case GLI_15:  xj = xi15; wj = w15; break;
	case GLI_20:  xj = xi20; wj = w20; break;
	case GLI_30:  xj = xi30; wj = w30; break;
	case GLI_40:  xj = xi40; wj = w40; break;
	case GLI_50:  xj = xi50; wj = w50; break;
	case GLI_60:  xj = xi60; wj = w60; break;
	case GLI_100:  xj = xi100; wj = w100; break;
	default:
		fprintf(stderr, "\t error in func.\"%s()\"\n", __func__);
		fprintf(stderr, "\t GLInum = %d \n", G_NUM);
		exit(EXIT_FAILURE);
	}
}
template<typename T>
void GaussIntegral<T>::Gauss_ForClass(vector<dbl>& x, vector<dbl>& w) {
	x = xj; w = wj;
}
template<typename T>
int GaussIntegral<T>::ShowNum() {
	return NUM;
}

template<typename T>
//T GaussIntegral<T>::GaussIntegralFunc(dbl init, dbl end, T (*func)(dbl s)) {
T GaussIntegral<T>::GaussIntegralFunc(dbl init, dbl end, function<T(dbl)> func) {
	T ret;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func((end - init) * xj[i] / 2.0 + (init + end) / 2.0);
	}
	ret *= (end - init) / 2.0;
	return ret;
}

template<typename T>
T GaussIntegral<T>::GaussIntegralFunc2Dmarginal1(dbl x_init, dbl x_end, dbl y, function<dbl(dbl, dbl)> func) {
	T ret;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func((x_end - x_init) * xj[i] / 2.0 + (x_init + x_end) / 2.0, y);
	}
	ret *= (x_end - x_init) / 2.0;
	return ret;
}

template<typename T>
T GaussIntegral<T>::GaussIntegralFunc2Dmarginal2(dbl x, dbl y_init, dbl y_end, function<dbl(dbl, dbl)> func) {
	T ret;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func(x, (y_end - y_init) * xj[i] / 2.0 + (y_init + y_end) / 2.0);
	}
	ret *= (y_end - y_init) / 2.0;
	return ret;
}

dbl GaussIntegral < dbl > ::GaussIntegralFunc2Dmarginal1(dbl x_init, dbl x_end, dbl y, function<dbl(dbl, dbl)> func) {
	dbl ret = 0.0;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func((x_end - x_init) * xj[i] / 2.0 + (x_init + x_end) / 2.0, y);
	}
	ret *= (x_end - x_init) / 2.0;
	return ret;
}

dbl GaussIntegral < dbl > ::GaussIntegralFunc2Dmarginal2(dbl x, dbl y_init, dbl y_end, function<dbl(dbl, dbl)> func) {
	dbl ret = 0.0;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func(x, (y_end - y_init) * xj[i] / 2.0 + (y_init + y_end) / 2.0);
	}
	ret *= (y_end - y_init) / 2.0;
	return ret;
}



dbl GaussIntegral<dbl>::GaussIntegralFunc(dbl init, dbl end, function<dbl(dbl)> func) {
	dbl ret = 0.0;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func((end - init) * xj[i] / 2.0 + (init + end) / 2.0);
	}
	ret *= (end - init) / 2.0;
	return ret;
}


Vector3d GaussIntegral<Vector3d>::GaussIntegralFunc(dbl init, dbl end, function<Vector3d(dbl)> func) {
	Vector3d ret;
	VectorXd v = VectorXd::Zero(3);
	ret = v;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func((end - init) * xj[i] / 2.0 + (init + end) / 2.0);
	}
	ret *= (end - init) / 2.0;
	return ret;
}

Vector2d GaussIntegral<Vector2d>::GaussIntegralFunc(dbl init, dbl end, function<Vector2d(dbl)> func) {
	Vector2d ret;
	VectorXd v = VectorXd::Zero(2);
	ret = v;
	for (int i = 0; i < NUM; i++) {
		ret += wj[i] * func((end - init) * xj[i] / 2.0 + (init + end) / 2.0);
	}
	ret *= (end - init) / 2.0;
	return ret;
}



