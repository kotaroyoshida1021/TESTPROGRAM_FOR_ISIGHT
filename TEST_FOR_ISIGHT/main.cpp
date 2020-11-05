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
static const int Kdim = NDIV - 2;

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
static dbl omegaXi_LL(dbl s) { return 0.0; }
static dbl omegaEta_LL(dbl s) { return 2.917010; }
static dbl omegaZeta_LL(dbl s) { return 0.0; }
static dbl kappa(dbl s) { return sqrt(Square(omegaXi_LL(s)) + Square(omegaEta_LL(s))); }
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

static void initializing() {
	std::cout << "Initializing...";
	calcWireLine();
//積分の区間数設定：スカラー
	cout << "SetGaussianParameters...";
	VecIntergalFunc.SetGaussIntegralParams(GLI_30);
//積分の区間数設定：ベクトル
	ScalarIntegralFunc.SetGaussIntegralParams(GLI_30);
	cout << "done\n";
//得られたハイパーパラメータを読み込む関数
	FILE* fp;
	cout << "Initializing HyperParameters...";
	string fname = "HyperParams.txt";
	errno_t err = fopen_s(&fp, fname.c_str(), "r");
	dbl tmp;
	vector<dbl> h_params;
	if (err) {
		cout << "File Open Error; " + fname << "\n";
		exit(1);
	}
	while (fscanf_s(fp, "%le ", &tmp) != EOF) {
		h_params.push_back(tmp);
		//cout << tmp << "\n";
	}
	cout << "done\n";
	VectorXd HyperParams = Map<VectorXd>(&h_params[0],h_params.size());
	AlphaParams = VectorXd::Zero(3);
	OmgEtaParams = VectorXd::Zero(3);
	DistParams = VectorXd::Zero(3);
//ハイパーパラメータを分ける
	for (int i = 0; i < h_params.size() / 3; i++) {
		OmgEtaParams(i) = HyperParams(i);
		DistParams(i) = HyperParams(3 + i);
		AlphaParams(i) = HyperParams(i + 3 * 2);
	}
	fclose(fp);
//Inputファイルから係数を読み出す
//初期化
	BETA = vector<dbl>(NDIV);
	OMG_ETA = vector<dbl>(NDIV);
	//coef->initializing
	cout << "all done\n";
}

static dbl RadiusBasisFunc(dbl s_i, dbl s_j, VectorXd Params) { return Params(0) * exp(-0.5 * Square(s_i - s_j) / (Params(1) * Params(1))); }
static dbl RadiusBasisFuncSdot(dbl s, dbl s_j, VectorXd Params) { return -Params(0) * (s - s_j) * exp(-0.5 * Square(s - s_j) / Params(1) * Params(1)) / Square(Params(1)); }
static void divideCoef(VectorXd a) {
	int i;
	for (i = 0; i < NDIV; i++) {
		BETA[i] = a[i];
	}
	Xi0Vec(0) = a[NDIV];
	Xi0Vec(0) = a[NDIV+1];
	OMG_ETA0 = 2 * atan(a[NDIV+2]) / M_PI;
	OMG_ETA_COEF = (kappa(0) - OMG_ETA0) / (kappa(0) + OMG_ETA0);
}

static dbl IntegrandForOmgEta(dbl s) { return 2.0 * kappa(s) * beta(s); }
static dbl IntegrateForOmgEta(dbl s) { return ScalarIntegralFunc.GaussIntegralFunc(0.0, s, bind(&IntegrandForOmgEta, _1)); }
static dbl omgEta(dbl s) { return kappa(s) * (1 - OMG_ETA_COEF*exp(IntegrateForOmgEta(s)) / (1 + OMG_ETA_COEF*exp(IntegrateForOmgEta(s)))); }
static dbl omgXi(dbl s) { return DiffSqrt(kappa(s), omgEta(s)); }
static dbl omgZeta(dbl s) { return -omgXi(s) * beta(s); }

static Coordinates obj_L(NDIV, length_LL, omgXi,omgEta,omgZeta);
static void initializeForCalcObj(VectorXd a) {
	divideCoef(a);
	beta.set_Info(Ds, BETA);
}
//目的関数の被積分関数
static dbl objective_integrand(dbl s) {
	cout << "Now..." << __func__;
	cout << "...s->" << s << "\n";
	Vector3d zetaSdot = omegaEta_LL(s) * obj_LL.xi(s) - omegaXi_LL(s) * obj_LL.eta(s);
	if (isnan(Square(obj_LL.zeta(s).dot(obj_L.zeta(s)) - 1)) || isnan(Square(zetaSdot.dot(obj_L.xi(s)) - omgEta(s)))) {
		cout << "isNan is found! condition>>";
		cout << "param = " << s << ", zeta = " << obj_LL.zeta(s).dot(obj_L.zeta(s)) - 1 << ", OmgEta = " << zetaSdot.dot(obj_L.xi(s)) - omgEta(s) << "\n";
		exit(1);
	}
	return Square(obj_LL.zeta(s).dot(obj_L.zeta(s)) - 1) + Square((zetaSdot.dot(obj_L.xi(s))-omgEta(s))/kappa(s));
}

static dbl objective(int n,VectorXd &a) {
#ifdef MY_DEBUG_MODE
	cout << "calculate objective...";
#endif
	initializeForCalcObj(a);
	Vector3d xi0, eta0, zeta0;
	zeta0 = obj_LL.ZETA[0];
	for (int i = 0; i < 2; i++) {
		xi0(i) = Xi0Vec(i);
	}
	xi0(2) = -(xi0(0) * zeta0(0) + xi0(1) * zeta0(1)) / zeta0(2);
	xi0.normalize();
	eta0 = zeta0.cross(xi0);
	obj_L.DetermineAxies(xi0, eta0, zeta0);
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
	return ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, bind(&objective_integrand, _1));
}


//逆行列微分のためのサブモジュールの定義
static MatrixXd calcGramMatrix(VectorXd Params) {
	MatrixXd Mat = MatrixXd::Zero(Kdim, Kdim);
	for (int i = 1; i < NDIV-1; i++) {
		for (int j = 1; j < NDIV-1; j++) {
			Mat(i - 1, j - 1) = RadiusBasisFunc(i * Ds, j * Ds, Params);
		}
	}
	Mat += MatrixXd::Identity(Kdim, Kdim) * Params[2];
	//行列がフルランクになるまで，仮想的な誤差項を足し合わせる．
	MatrixXd EPS = 1.0e-4 * (MatrixXd::Identity(Kdim, Kdim));
	MatrixXd MAT_TMP = Mat;
	int flag = 0;
	for (int i = 0; i < 3; i++) {
		FullPivLU<MatrixXd> lu_decomp(MAT_TMP);
		int rank = lu_decomp.rank();
		if (rank != Kdim) {
			printf("rank is not\n");
			MAT_TMP = Mat + EPS;
			EPS *= 10.0;
			flag = 1;
		}
	}
	if (flag == 1) {
		Mat = MAT_TMP;
	}
	else {
		Mat = Mat + EPS;
	}
	return Mat;
	
}

static MatrixXd PartialKernelMatrix(int NUM,VectorXd Params) {
	MatrixXd pK;
	MatrixXd I = MatrixXd::Identity(Kdim, Kdim);
	pK = MatrixXd::Identity(Kdim, Kdim);
	if (NUM != 2) {
		for (int i = 1; i < NDIV-1; i++) {
			for (int j = 1; j < NDIV-1; j++) {
				switch (NUM) {
				case 0:
					pK(i-1, j-1) = RadiusBasisFunc(i * Ds, j * Ds, Params) / Params(0);
				case 1:
					pK(i-1, j-1) = (Square(i * Ds - j * Ds)) / (Square(Params(1))* Square(Params(1))) * RadiusBasisFunc(i * Ds, j * Ds, Params);
				case 2:
					//pK[i, j] = Params(0) * exp(Square(i * Ds - j * Ds) / Params(1));
					break;
				default:
					cout << "Error in func;" << __func__ << " -> SwitchError; i=" << i << "\n";
					exit(1);
				}
			}
		}
	}
	
	return pK;
}

static void PreCalcForConds() {
	MatrixXd Kinv,K,tmp,K_i;
	K = calcGramMatrix(AlphaParams);
	Kinv = K.inverse();
	for (int i = 0; i < 3; i++) {
		K_i = PartialKernelMatrix(i, AlphaParams);
		tmp = Kinv * K_i;
		detPartDiff[i] = tmp.trace();
		KinvPartdiff[i] = tmp * Kinv;
	}
	K = calcGramMatrix(OmgEtaParams);
	Kinv = K.inverse();
	for (int i = 0; i < 3; i++) {
		K_i = PartialKernelMatrix(i, OmgEtaParams);
		tmp = Kinv * K_i;
		detPartDiff[i+3] = tmp.trace();
		KinvPartdiff[i+3] = tmp * Kinv;
	}
	K = calcGramMatrix(DistParams);
	Kinv = K.inverse();
	for (int i = 0; i < 3; i++) {
		K_i = PartialKernelMatrix(i, DistParams);
		tmp = Kinv * K_i;
		detPartDiff[i + 3 * 2] = tmp.trace();
		KinvPartdiff[i + 3 * 2] = tmp * Kinv;
	}
	for (int i = 0; i < 9; i++) {
		if (isnan(detPartDiff[i])) {
			printf("isnan is detPartDiff-> i=%d, f = %f", i, detPartDiff[i]);
			exit(1);
		}
	}
}

static void CalcConds(int n, VectorXd& a, int ncond, VectorXd& COND) {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "\n";
#endif
	divideCoef(a);
	vector<dbl> d;
	VectorXd A, E;
	A = VectorXd::Zero(NDIV - 2);
	E = VectorXd::Zero(NDIV - 2);
	for (int i = 1; i < NDIV - 1; i++) {
		A[i - 1] = atan(BETA[i]);
		E[i - 1] = OMG_ETA[i];
	}
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
	cout << "now...";
	for (int i = 0; i < 3; i++) {
		d.push_back(-detPartDiff[i] + A.dot(KinvPartdiff[i] * A));
	}
	cout << "alpha...";
	for (int i = 0; i < 3; i++) {
		d.push_back(-detPartDiff[i + 3] + E.dot(KinvPartdiff[i + 3] * E));
	}
	cout << "eta..";
	if (d.size() != ncond) {
		cout << "size not match\n";
		exit(1);
	}
	for (int i = 0; i < d.size(); i++) {
		if (isnan(d[i])) {
			printf("isnan is Detected -> i = %d, f=%f", i, d[i]);
			exit(1);
		}
	}

	cout << "all done\n";
	COND = Eigen::Map<Eigen::VectorXd>(&d[0], d.size());
	//COND *= 5.0;
	vector<dbl>().swap(d);		
}

static dbl DistMax(dbl s) {
	//return cos(atan(beta(s)) / (alphaSdot(s) + omgEta(s));
}

static vector<dbl> calcIneqs() {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__;
#endif
	vector<dbl> I;
	return I;
#ifdef MY_DEBUG_MODE
	cout << "all done\n";
#endif
}

int main(int argc, char** argv)
{
	initializing();

	ofstream OBJ, COND, INEQ;
	OBJ.open("objfunc.txt");
	COND.open("conds.txt");
	INEQ.open("ineqs.txt");
	PreCalcForConds();
	for (int i = 0; i < 9; i++) {
		printf("%f\n",detPartDiff[i]);
	}
	vector<dbl> c, i;
	VectorXd CONDS;
	coef = VectorXd::Ones(NCOORD);
	CalcConds(NCOORD,coef,NCOND,CONDS);
	for (int p = 0; p < 6; p++) {
		//COND.write("%lf", c[p]);
		COND << CONDS(p);
		if (p != 6) COND << ",";
		cout << "i = " << p << "\n";
	}
	COND.close();
	i = calcIneqs();
	OBJ << objective(NCOORD,coef);
	
	
}