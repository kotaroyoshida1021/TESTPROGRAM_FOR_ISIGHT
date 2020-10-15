#include "pch.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <Eigen/Core>
#include <functional>
#include <random>
#include <direct.h>
#include <numeric>
#include "LinearFunction.h"
#include "GaussIntegral.h"
#include "Obj_Coordinates.h"
#include "Nelder_Mead.h"
#include "MultiSimp.h"
#include "gnuplot.h"
#include "Ritz.h"
#include "IntervalReduction.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "simparams.h"
#include "SurfaceInfo.h"
#include "Timer.h"

using namespace std;
using namespace Eigen;
using namespace placeholders;

static inline dbl Square(dbl f) {
	return f * f;
}
static void swap(dbl &x, dbl &y) {
	dbl tmp = y;
	y = x;
	x = tmp;
}
static inline dbl DiffSqrt(dbl x, dbl y) {
	if (x < y) {
		swap(x, y);
	}
	return sqrt(Square(x) - Square(y));
}
static dbl omegaXi_LL(dbl s) {
	return 0.0;
}
static dbl omegaEta_LL(dbl s) {
	return 2.917010;
}
static dbl omegaZeta_LL(dbl s) {
	return 0.0;
}
static dbl kappa(dbl s) {
	return sqrt(Square(omegaXi_LL(s)) + Square(omegaEta_LL(s)));
}
static Coordinates obj_LL(NDIV, length_LL, omegaXi_LL, omegaEta_LL, omegaZeta_LL);
//ワイヤーの計算
static void calcWireLine(void) {
	dbl chi, delta;
	chi = 0.0; //delta = 0.343004 - M_PI / 2.0;
	delta = -M_PI / 2.0;
	Vector3d Xi0, Eta0, Zeta0;
	Xi0 << cos(delta), 0.0, -sin(delta);
	Eta0 << sin(chi) * sin(delta), cos(chi), sin(chi)* cos(delta);
	Zeta0 << cos(chi) * sin(delta), -sin(chi), cos(chi)* cos(delta);
	obj_LL.DetermineAxies(Xi0, Eta0, Zeta0);
}
static void initializing(string InputFname) {
	std::cout << "Initializing...";
	calcWireLine();
//積分の区間数設定：スカラー
	VecIntergalFunc.SetGaussIntegralParams(GLI_30);
//積分の区間数設定：ベクトル
	ScalarIntegralFunc.SetGaussIntegralParams(GLI_30);
//得られたハイパーパラメータを読み込む関数
	FILE* fp;
	string fname = "HyperParams.txt";
	errno_t err = fopen_s(&fp, fname.c_str, "r");
	dbl tmp;
	vector<dbl> h_params;
	if (err) {
		cout << "File Open Error; " + fname << "\n";
		exit(1);
	}
	while (fscanf_s(fp, "%le\n", &tmp) != EOF) {
		h_params.push_back(tmp);
	}
	VectorXd HyperParams = VectorXd(h_params);
	AlphaParams = VectorXd::Zero(h_params.size() / 3);
	OmgEtaParams = VectorXd::Zero(h_params.size() / 3);
	DistParams = VectorXd::Zero(h_params.size() / 3);
//ハイパーパラメータを分ける
	for (int i = 0; i < h_params.size() / 3; i++) {
		AlphaParams(i) = HyperParams(i);
		OmgEtaParams(i) = HyperParams(h_params.size() / 3 + i);
		DistParams(i) = HyperParams(h_params.size() * 2 / 3 + i);
	}
	fclose(fp);
//Inputファイルから係数を読み出す
	errno_t err = fopen_s(&fp, InputFname.c_str, "r");
	vector<dbl> InputCoef;
	if (err) {
		cout << "File Open Error; " + InputFname << "\n";
		exit(1);
	}
	while (fscanf_s(fp, "%lf\n", &tmp) != EOF) {
		InputCoef.push_back(tmp);
	}
	if (dim != (InputCoef.size()-2) / 3) {
		cout << "Size not match: InputCoef.size() = " << InputCoef.size() / 3 << " dim = " << dim << "\n";
		exit(1);
	}
	for (int i = 0; i < dim; i++) {
		AlphaCoef(i) = InputCoef[i];
		OmgEtaCoef(i) = InputCoef[i + dim];
		DistCoef(i) = InputCoef[i + 2 * dim];
	}
	Xi0Vec(0) = InputCoef[3 * dim];
	Xi0Vec(1) = InputCoef[3 * dim + 1];
	//coef->initializing
	S = VectorXd::Zero(dim);
	for (int i = 0; i < dim; i++) {
		S(i) = i * length_LL / (dim - 1);
	}
	cout << "done\n";
}

//再生核ヒルベルト空間っぽい表現にする
static dbl alpha(dbl s) {
	VectorXd kernel = VectorXd::Zero(dim);
	for (int i = 0; i < dim; i++) {
		kernel(i) = AlphaParams(0) * exp(0.5 * (s - S(i)) * (s - S(i)) / (AlphaParams[1] * AlphaParams[1]));
	}
	return AlphaCoef.dot(kernel);
}

static dbl omgEta(dbl s) {
	VectorXd kernel = VectorXd::Zero(dim);
	for (int i = 0; i < dim; i++) {
		kernel(i) = OmgEtaParams(0) * exp(0.5 * (s - S(i)) * (s - S(i)) / (OmgEtaParams[1] * OmgEtaParams[1]));
	}
	return OmgEtaCoef.dot(kernel);
}

static dbl dist(dbl s) {
	VectorXd kernel = VectorXd::Zero(dim);
	for (int i = 0; i < dim; i++) {
		kernel(i) = DistParams(0) * exp(0.5 * (s - S(i)) * (s - S(i)) / (DistParams[1] * DistParams[1]));
	}
	return DistCoef.dot(kernel);
}

static dbl omgXi(dbl s) {
	return DiffSqrt(omgXi(s), kappa(s));
}

static dbl omgZeta(dbl s) {
	return -omgXi(s) * tan(alpha(s));
}

static Coordinates obj_L(NDIV, length_LL, omgXi, omgEta, omgZeta);

static dbl objective_integrand(dbl s) {
	Vector3d zetaSdot = omegaEta_LL(s) * obj_LL.xi(s);
	return Square(obj_LL.zeta(s).dot(obj_L.zeta(s)) - 1) + Square(zetaSdot.dot(obj_L.xi(s))-omgEta(s));
}
static dbl objective(VectorXd &a) {
	Vector3d xi0, eta0, zeta0;
	zeta0 = obj_LL.ZETA[0];
	for (int i = 0; i < 2; i++) {
		xi0(i) = Xi0Vec(i);
	}
	xi0(2) = -(xi0(0) * zeta0(0) + xi0(1) * zeta0(1)) / zeta0(2);
	xi0.normalize();
	eta0 = zeta0.cross(xi0);
	obj_L.DetermineAxies(xi0, eta0, zeta0);
	return ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, bind(&objective_integrand, _1));
}
//逆行列微分のためのサブモジュールの定義
static MatrixXd PartialKernelMatrix(int NUM,VectorXd Params) {
	MatrixXd pK = MatrixXd::Zero(NDIV - 2, NDIV - 2);
	for (int i = 1; i < NDIV - 1; i++) {
		for (int j = 1; j < NDIV - 1; j++) {
			switch (NUM) {
				case 0:
					pK[i, j] = exp(Square(i * Ds - j * Ds) / Params(1));
				case 1:
					pK[i, j] = -Params(0) * exp(Square(i * Ds - j * Ds) / Params(1)) * (Square(i * Ds - j * Ds)) / (Square(Params(1)));
				case 2:
					pK[i, j] = Params(0) * exp(Square(i * Ds - j * Ds) / Params(1));
				default:
					cout << "Error in func;" << __func__ << " -> SwitchError; i=" << i << "\n";
					exit(1);
			}
		}
	}
	
}

static void CalcConds(VectorXd& Cond) {
	vector<dbl> d;
}

int main(int argc, char** argv)
{
	string INPUT_FILENAME;
	bool FILE_MATCH = TRUE;
	INPUT_FILENAME = string(argv[1]);
	if (INPUT_FILENAME != "INPUT.txt") {
		cout << "Wrong File Name: Check Input file name and argument" << "\n";
		exit(1);
	}
	initializing(INPUT_FILENAME);
	string fname;
	string TEMP;
	TEMP = "./temp";
	_mkdir(TEMP.c_str());
	status stat;
	VectorXd OptVec;
	
}