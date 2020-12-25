#include "pch.h"
#include <iostream>
#include <iterator>
#include <filesystem>
#include <vector>
#include <Eigen/Eigen>
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
#include <nlopt.hpp>
#include <iomanip>

using namespace std;
using namespace Eigen;
using namespace placeholders;
using namespace nlopt;

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
	if (x == 0.0) {
		return y;
	}
	else if(y==0.0){
		return x;
	}
	else {
		return sqrt(Square(x) - Square(y));
	}	
}

static dbl omegaXi_LL(dbl s) { return 0.0; }
static dbl omegaEta_LL(dbl s) { return 2.917010; }
static dbl omegaZeta_LL(dbl s) { return 0.0; }
static dbl kappa(dbl s) { return sqrt(Square(omegaXi_LL(s)) + Square(omegaEta_LL(s))); }
static Coordinates obj_LL(NDIV, length_LL, omegaXi_LL, omegaEta_LL, omegaZeta_LL);

//ワイヤーの計算:
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

//INITIALIZING->INITIAL PARAMETERS FOR CALCULATION
static void initializing() {
	std::cout << "Initializing...";
	calcWireLine();
//SET GAUSSIAN INTEGRAL PARAMETERS OF SCALAR 
	cout << "SetGaussianParameters...";
	VecIntergalFunc.SetGaussIntegralParams(GLI_30);
//SET GAUSSIAN INTEGRAL PARAMETERS OF VECTOR
	ScalarIntegralFunc.SetGaussIntegralParams(GLI_30);
	cout << "done\n";
//READ HYPERPARAMETERS FROM INPUT FILE
	FILE* fp;
	cout << "Initializing HyperParameters...";
	string fname = "HyperParams3.txt";
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
//DIVIDE PARAMS
	for (int i = 0; i < 3; i++) {
		OmgEtaParams(i) = h_params[i];
		DistParams(i) = h_params[i + 3];
		AlphaParams(i) = h_params[i + 3 * 2];
	}
	NCOORD_ALPHA = NDIV;
	NCOORD_ETA = NDIV;
	NCOORD = NCOORD_ALPHA + NCOORD_ETA + 2;
	cout << "NCOORD = " << NCOORD << "\n";
	fclose(fp);
	cout << "all done\n";

}
MatrixXd calcGramMatrix(VectorXd Params);
static void memorizeGramMatrix() {
	cout << "now " << __func__ << "...";
	GramA = calcGramMatrix(AlphaParams);
	GramE = calcGramMatrix(OmgEtaParams);
	GramD = calcGramMatrix(DistParams);
}
//Determine the dimension of coefficient for optimizing使わね
static void determineDimension() {
	FullPivLU<MatrixXd> lu_decompA(GramA), lu_decompE(GramE), lu_decompD(GramD);
	cout << "rank = " << lu_decompA.rank() << ", " << lu_decompE.rank() << ", " << lu_decompD.rank() << "...";
	RankA = lu_decompA.rank();
	RankE = lu_decompE.rank();
	RankD = lu_decompD.rank();
	FILE* fp;
	errno_t err = fopen_s(&fp, "EIGVEC_MAX.txt", "r");
	//vector<dbl> EigVal;
	dbl tmp;
	while (fscanf_s(fp, "%le\n", &tmp) != EOF) {
		EigVal.push_back(tmp);
	}
	SelfAdjointEigenSolver<MatrixXd> ES;
	ES.compute(GramD);
	cout << ES.eigenvalues() << "\n";
	MatrixXd EigE = OmgEtaParams(2) * MatrixXd::Identity(Kdim,Kdim) - GramE;
	KerE = EigE.fullPivLu().kernel();
	NCOORD_ETA = EigE.fullPivLu().dimensionOfKernel();
	//Determine ALPHA DIMENSION;
	MatrixXd Eig = AlphaParams(2) * MatrixXd::Identity(Kdim, Kdim) - GramA;
	FullPivLU<MatrixXd> lu_decompEig(Eig);
	KerA = lu_decompEig.kernel();
	NCOORD_ALPHA = Kdim - lu_decompEig.rank();
	//Determine DIST DIMENSION;
	MatrixXd EigD = DistParams(2) * MatrixXd::Identity(Kdim, Kdim) - GramD;
	FullPivLU<MatrixXd> lu_decompEigD(EigD);
	KerD = lu_decompEigD.kernel();
	NCOORD_DIST = Kdim - lu_decompEigD.rank();

	PARAM_E = VectorXd::Zero(NCOORD_ETA);
	PARAM_D = VectorXd::Zero(NCOORD_DIST);
	PARAM_A = VectorXd::Zero(NCOORD_ALPHA);
	cout << "KerSize->" << KerA.size() / Kdim << ", " << KerE.size() / Kdim << ", " << KerD.size() / Kdim << "...";
	NCOORD_DIST = 0;
	NCOORD_ETA = 0;
	NCOORD = NCOORD_ALPHA + NCOORD_DIST + NCOORD_ETA + 3;
	//NCOORD = NCOORD_ALPHA + NCOORD_ETA + 2;
	cout << "NCOORD = " << NCOORD << "\n";
}

static dbl RadiusBasisFunc(dbl s_i, dbl s_j, VectorXd Params) { return Params(0) * exp(-0.5 * Square(s_i - s_j) / (Params(1) * Params(1))); }
static dbl RadiusBasisFuncSdot(dbl s, dbl s_j, VectorXd Params) { return -Params(0) * (s - s_j) * exp(-0.5 * Square(s - s_j) / Params(1) * Params(1)) / Square(Params(1))* Params(1); }
/*
	!! MUST USE IN FUNCTIONS WHICH ARE ARGUMENT OF NLOPT !!
	USAGE : DIVIDE INPUT VECTORS TO OPTIMIZIED FUNCTION
*/
#define atanSigmoid(s) ((atan(s)*M_2_PI))
#define sinarctanSigmoid(s)((s/sqrt(Square(s)+1.0)))
static void divideCoef(VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << a.size();
	cout << "now " << __func__ << "...";
#endif
	for (int i = 0; i < NCOORD - 3; i++) {
		if (i < NCOORD_ALPHA) {
			PARAM_A(i) = a(i);
		}
		else if (i >= NCOORD_ALPHA && i < NCOORD_ALPHA + NCOORD_ETA) {
			PARAM_E(i - NCOORD_ALPHA) = a(i);
		}
		else {
			PARAM_D(i - NCOORD_ALPHA+NCOORD_ETA) = a(i);
		}
	}
	//cout << "\n";
	OMG_0 = -1 + exp(a[NCOORD - 3]);
	Xi0Vec(0) = a[NCOORD - 2];
	Xi0Vec(1) = a[NCOORD - 1];
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
}
/*
	THESE WERE FUNCTIONS FOR LINAER INTERPOLATE, EXPRESSED BY REPRODUSING KERNEL HILBERT SPACE 
*/

static dbl EigenFunc(VectorXd PARAM, dbl s){
	if (PARAM.size() != Kdim) {
		cout << "error size not match" << "->" << PARAM.size() << "\n";
		exit(1);
	}
	else {
		if (s < Ds) {
			return PARAM(0);
		}
		else if (s > length_LL - Ds && s <= length_LL) {
			return PARAM(Kdim - 1);
		}
		else {
			int n = s / Ds;
			dbl p = s - n * Ds;
			if (p == 0.0) {
				return PARAM(n - 1);
			}
			else {
				return (1 - p) * PARAM(n - 1) + p * PARAM(n);
			}
		}
	}
}

static dbl Alpha(dbl s) {
	return EigenFunc(BETA_H, s);
}

static dbl Beta(dbl s) { return tan(Alpha(s)); }
static dbl forOMG_INTEGRAND(dbl s) { return 2 * kappa(s) * Beta(s); }

static dbl omgEta(dbl s) {
	//return EigenFunc(OMG_ETA_H, s);
	dbl p = ScalarIntegralFunc.GaussIntegralFunc(0.0, s, forOMG_INTEGRAND);
	return kappa(s) * (OMG_0 * exp(p) - 1) / (OMG_0 * exp(p) + 1);
}

static dbl DistMax(dbl s) {
	return sqrt(1 + beta(s) * beta(s)) / (beta.derivative(s) + (1 + beta(s) * beta(s)) * omgEta(s));
}
#define SIGMOID(s) (1.0/(1.0+exp(-s)))
static dbl Dist(dbl s) {
	if (s == 0.0 || s == length_LL) {
		return 0.0;
	}
	else {
		return DistMax(s) * SIGMOID(forDIST.Function(s));
	}
}
/*
	THESE WERE FUNCTIONS FOR OBJECTIVE AND CONDITIONS
*/
static dbl omegaEtaWrap(dbl s) { return omgEta(s); }
static dbl phi(dbl s) { return acos(omgEta(s)/kappa(s)); }
static dbl omegaXi(dbl s) { return kappa(s) * sin(phi(s)); }
static dbl omegaZeta(dbl s) { return -omegaXi(s) * Beta(s); }


static inline void MemorizeFunctions() {
#ifdef MY_DEBUG_MODE
	cout << "Memorize..";
#endif
	OMG_ETA_H = KerE * PARAM_E;
	BETA_H = KerA * PARAM_A;
	//ALPHA = vector<dbl>(Kdim);
	for (int i = 1; i < Kdim+1; i++) {
		ALPHA[i - 1] = Alpha(i * Ds);
	}

#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
}
static Coordinates obj_L(NDIV, length_LL, omegaXi,omegaEtaWrap,omegaZeta);
//FOR CALCULATE BENDING ENERGY
static dbl alphaSdot(dbl s) {
	dbl tmp = 0.0;
	for (int i = 0; i < Kdim; i++) {
		tmp += forBt(i) * RadiusBasisFuncSdot(s, (i + 1) * Ds, AlphaParams);
	}
	return tmp;
}
//PROTOTYPE DECREARE
static dbl IntegrandForCond_Zeta(dbl s);
static dbl IntegrandForCond_Pos(dbl s);
static dbl IntegrandForCond_omgEta(dbl s);
static dbl BarrierFunc(dbl s) {
	//return 1 / Square(sin(phi(s)));
	//return 1 / Square(omegaEta(s) - kappa(s));
	Vector3d e_y = Vector3d::Unit(1);
	dbl ALPH = atan(beta(s));
	dbl ret = -obj_L.zeta(s).dot(e_y) * sin(ALPH) + obj_L.xi(s).dot(e_y) * cos(ALPH);
	if (ret < 0) {
		return 1.0e12;
	}
	else {
		return -log(fabs(-obj_L.zeta(s).dot(e_y) * sin(ALPH) + obj_L.xi(s).dot(e_y) * cos(ALPH)));
	}
}//-log(fabs(sin(phi(s)))); }
static dbl objective_integrand(dbl s) {
#ifdef MY_DEBUG_MODE
	cout << "Now..." << __func__;
	cout << "...s->" << s << "\n";
#endif
	//return IntegrandForCond_omgEta(s) + IntegrandForCond_Pos(s) + IntegrandForCond_Zeta(s);// + BarrierFunc(s);
	return IntegrandForCond_omgEta(s) + IntegrandForCond_Pos(s) + IntegrandForCond_Zeta(s);
	//return IntegrandForCond_Pos(s) + IntegrandForCond_Zeta(s); +BarrierFunc(s);

}
static inline void initializeForCalcConds(VectorXd a);
void initializeForCalcObj(VectorXd a) {
	//divideCoef(a);
	//MemorizeFunctions();
	initializeForCalcConds(a);
}

static bool CheckExceedOmgEta(VectorXd a) {
	bool c = false;
	divideCoef(a);
	MemorizeFunctions();
	for (int i = 0; i < NDIV; i++) {
		if (omgEta(i*Ds) > kappa(i*Ds)) {
			c = true;
		}
	}

	return c;
}

static void CalcConds(int n, VectorXd& a, int ncond, vector<dbl>& COND);
static dbl objective(int n, VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << "calculate objective...";
#endif
	//initializeForCalcObj(a);
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
	if (CheckExceedOmgEta(a)) {
		return 1.0e+9;
	}
	else {
		vector<dbl> NIZI;
		//CalcConds(NCOORD, a, NCOND, NIZI);
		initializeForCalcObj(a);
		dbl ret = ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, bind(&objective_integrand, _1)) + AlphaParams(2)*PARAM_A.lpNorm<1>();
		//dbl ret = Square(NIZI[0]) + Square(NIZI[1])  + ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, BarrierFunc);
		obj_L.terminate();
		return ret;
	}
	
}

static inline void initializeForCalcConds(VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "...";
#endif
	divideCoef(a);
	MemorizeFunctions();
	Vector3d xi0, eta0, zeta0;
	zeta0 = obj_LL.ZETA[0];
	if (zeta0(2) != 0.0) {
		for (int i = 0; i < 2; i++) {
			xi0(i) = Xi0Vec(i);
		}
		xi0(2) = -(xi0(0) * zeta0(0) + xi0(1) * zeta0(1)) / zeta0(2);
		xi0.normalize();
		eta0 = zeta0.cross(xi0);
	}
	else {
		for (int i = 1; i < 3; i++) {
			xi0(i) = Xi0Vec(i - 1);
		}
		//xi0(2) = -(xi0(0) * zeta0(0) + xi0(1) * zeta0(1)) / zeta0(2);
		xi0(0) = -(xi0(1) * zeta0(1) + xi0(2) * zeta0(2)) / zeta0(0);
		xi0.normalize();
		eta0 = zeta0.cross(xi0);
	}
	obj_L.DetermineAxies(xi0, eta0, zeta0);
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
}

static inline void finalizeForCalcConds() { obj_L.terminate(); }

static dbl IntegrandForCond_Zeta(dbl s) { return Square(obj_L.zeta(s).dot(obj_LL.zeta(s)) - 1); }
static dbl IntegrandForCond_Pos(dbl s) {
	VectorXd diff = obj_L.pos(s) - obj_LL.pos(s);
	return diff.norm();
}

static dbl IntegrandForCond_Pos_i(int i, dbl s) {
	VectorXd diff = obj_L.pos(s) - obj_LL.pos(s);
	return diff(i) * diff(i);
}

static dbl IntegrandForCond_omgEta(dbl s) {
	Vector3d zetaSdot = -omegaXi_LL(s) * obj_LL.eta(s) + omegaEta_LL(s) * obj_LL.xi(s);
	return Square((zetaSdot.dot(obj_L.xi(s)) - omgEta(s)) / kappa(s));
}

static MatrixXd calcGramMatrix(VectorXd Params) {
	MatrixXd Mat = MatrixXd::Zero(Kdim, Kdim);
	for (int i = 1; i < NDIV-1; i++) {
		for (int j = 1; j < NDIV-1; j++) {
			Mat(i - 1, j - 1) = RadiusBasisFunc(i * Ds, j * Ds, Params);
		}
	}
	Mat += MatrixXd::Identity(Kdim, Kdim) * Params[2];
	return Mat;
	
}

static void CalcConds(int n, VectorXd& a, int ncond, vector<dbl> &COND) {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "...";
#endif
	initializeForCalcConds(a);
	vector<dbl> d;
	VectorXd A, E;
	A = Eigen::Map<Eigen::VectorXd>(&ALPHA[0], ALPHA.size());
	E = VectorXd::Zero(Kdim);
	for (int i = 0; i < Kdim; i++) {
		E[i] = OMG_ETA[i + 1];
	}
	VectorXd a_A, a_E;
	a_A = GramA.fullPivLu().solve(A);
	a_E = GramE.fullPivLu().solve(E);
	//d.push_back((a_E.dot(E)/Square(E.norm()) - 1 / EigVal[0]));
	d.push_back(a_E.dot(E) / (a_E.norm() * a_E.norm()) - OmgEtaParams(2));
	//d.push_back((a_A.dot(A)/Square(A.norm()) -  1 / EigVal[1]));
	d.push_back(a_A.dot(A) / (a_A.norm() * a_A.norm()) - AlphaParams(2));
	for (int i = 0; i < 3; i++) {
		if (isnan(d[i])) {
			cout << "isNaN detected:-> I = " << i << "\n";
			exit(1);
		}
	}
	COND = d;
	/*
	if (d.size() != NCOND) {
		cout << "error in func :" << __func__ << " -> quantity of condition is not match" << "\n";
		exit(1);
	}
	*/
#ifdef MY_DEBUG_MODE
	cout << "done:" << __func__ << "\n";
#endif
	vector<dbl>().swap(d);
	finalizeForCalcConds();
}
void initializeForCalcIneq(VectorXd a) {
	divideCoef(a);
	MemorizeFunctions();
}
static void calcIneqs(int n, VectorXd &a,int nineq,vector<dbl> &INEQ) {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "...";
#endif
	vector<dbl> I;
	initializeForCalcIneq(a);
	for (int i = 0; i < NDIV; i++) {
		I.push_back(fabs(OMG_ETA[i]) - fabs(kappa(i * Ds)));
	}
	//for (int i = 0; i < NDIV; i++) {
		//I.push_back(DIST[i] - DistMax(i * Ds));
		//I.push_back(-DIST[i]);
	//}
	//INEQ = Eigen::Map<Eigen::VectorXd>(&I[0], I.size());
	//INEQ = I;
	INEQ = I;
	if (I.size() != NINEQ) {
		cout << "error in func :" << __func__ << " -> quantity of condition is not match" << "\n";
		exit(1);
	}
#ifdef MY_DEBUG_MODE
	cout << "all done\n";
#endif
	vector<dbl>().swap(I);
}
//NLopt用WrapperSeries

void fprint_for_gnu(string FILE_NAME,VectorXd a) {
	//Update_VectorData();
	initializeForCalcObj(a);
	int i;
	ofstream temp;
	ofstream input;
	temp.open(FILE_NAME, std::ios::out);
	for (i = 0; i < NDIV; i++) {
		//temp << obj_LL.POS[i](2) << " " << obj_LL.POS[i](0) << " " << obj_LL.POS[i](1) << "\n";
	}
	temp << "\n";
	for (i = 0; i < NDIV; i++) {
		//temp << obj_L.POS[i](2) << " " << obj_L.POS[i](0) << " " << obj_L.POS[i](1) << "\n";
		temp << i * Ds << " " << Alpha(i*Ds) << " " << omgEta(i * Ds) << "\n";
	}
	temp << "\n\n";
	//temp << "\n\n";
	//for (i = 0; i < NDIV; i++) {
		//temp << obj_U.POS[i](2) << " " << obj_U.POS[i](0) << " " << obj_U.POS[i](1) << "\n";
	//}
	temp.close();
	obj_L.terminate();
}
//FUNCTION WRAPPERS -> SHOULD WE NOT USE GRADIENT
int EVAL_COUNTER = 0;
//unsigned m, double* result, unsigned n, const double* x,
//double* gradient, /* NULL if not needed */
//void* func_data
static void MultiCondFuncWrapper(unsigned m_cond, double *ret, unsigned ndims, const double* x,double *gradient,void* my_func_data) {
	
	dbl* a;
	//a = (dbl*)malloc(NCOORD);
	a = alloc(dbl, NCOORD);
	for (int i = 0; i < NCOORD; i++) {
		a[i] = x[i];
	}
#ifdef MY_DEBUG_MODE
	for (int i = 0; i < NCOORD; i++) {
		cout << a[i];
	}
#endif
	vector<dbl> tmp(a, a + NCOORD);
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	vector<dbl> COND(NCOND);
	//CalcConds(NCOORD, coef, NCOND, COND);
#ifdef MY_DEBUG_MODE
	cout << COND.size();
#endif
	for (int i = 0; i < NCOND; i++) {
		ret[i] = COND[i];
	}
	free(a);
}

static inline void MultiIneqFuncWrapper(unsigned m_cond, double* ret, unsigned ndims, const double* x, double* gradient, void* my_func_data) {
	//vector<dbl> tmp;
	dbl* a;
	//a = (dbl*)malloc(NCOORD);
	a = alloc(dbl, NCOORD);
	for (int i = 0; i < NCOORD; i++) {
		a[i] = x[i];
	}
	//copy(tmp.begin(), tmp.end(), a);
	vector<dbl> tmp(a, a + NCOORD);
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	vector<dbl>COND(NINEQ);
	calcIneqs(NCOORD, coef, NINEQ, COND);
	for (int i = 0; i < NINEQ; i++) {
		ret[i] = COND[i];
	}
	free(a);
}

static dbl Equality3(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data);
static dbl Equality4(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data);
static dbl Equality5(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data);
static dbl Equality6(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data);
static dbl Equality7(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data);
static dbl Equality8(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data);

static dbl objectiveWrapper(const vector<dbl> &x, vector<dbl> &grad, void *my_func_data) {
	
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	string F = file_name + "_tmp.txt";
	dbl ret = objective(NCOORD, coef);
	fprint_for_gnu(F, coef);
	if (!grad.empty()) {
		for (int i = 0; i < grad.size(); i++) {
			grad[i] = 0.0;
		}
	}
	ostringstream ss, si;
	ss << EVAL_COUNTER;
	if (EVAL_COUNTER == 0||EVAL_COUNTER % 100 == 0) {
		//fprintf_s(stderr, "\r%8d | %6.3e \r", EVAL_COUNTER, ret);
		cout << "\rEVAL_COUNTER = " << scientific << setprecision(5) << EVAL_COUNTER << " f = " << scientific << setprecision(5) << ret << ", ";
		//string command = "splot \"" + F + "\" u 1:2:3 w lp pt 2 t \"k = " + ss.str() + "\n";
		gp.Command("set y2tics");
		string command = "plot \"" + F + "\" u 1:2 w l axis x1y1 t \"k = " + ss.str() +" Beta\", \"" + F + "\" u 1:3 w l axis x1y2 t \"k = " + ss.str() + " omgEta \n";
		gp.Command(command.c_str());
		if (EVAL_COUNTER % 100 == 0) {
			vector<dbl> CON(NCOND);
			CON[0] = Equality3(x, grad, my_func_data);
			CON[1] = Equality4(x, grad, my_func_data);
			CON[2] = Equality5(x, grad, my_func_data);
			CON[3] = Equality6(x, grad, my_func_data);
			CON[4] = Equality7(x, grad, my_func_data);
			CON[5] = Equality8(x, grad, my_func_data);
			//CalcConds(NCOND, coef, NCOND, CON);
			for (int i = 0; i < CON.size(); i++) {
				cout << " C[" << i << "] = " << scientific << setprecision(5) << CON[i] << ", ";
			}
			//cout << ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_omgEta) << ", " << ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Pos) << ", " << ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Zeta);
			cout << "\r";
		}
	}
	++EVAL_COUNTER;
	return ret;
}

static dbl Equality1(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcConds(coef);
	vector<dbl> d;
	VectorXd E;
	//A = Eigen::Map<Eigen::VectorXd>(&ALPHA[0], ALPHA.size());
	E = VectorXd::Zero(Kdim);
	for (int i = 0; i < Kdim; i++) {
		E[i] = OMG_ETA[i + 1];
	}
	VectorXd a_A, a_E;
	//a_A = GramA.fullPivLu().solve(A);
	a_E = GramE.fullPivLu().solve(E);
	//d.push_back(a_E.dot(E) - 1 / EigVal[0]);
	//d.push_back(a_A.dot(A) - 1 / EigVal[1]);
	return a_E.dot(E)/a_E.dot(a_E) - 1 / EigVal[0];
	//return a_E.dot(E) - OmgEtaParams(2);
}

static dbl Equality2(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcConds(coef);
	VectorXd A;
	A = Eigen::Map<Eigen::VectorXd>(&ALPHA[0], ALPHA.size());
	VectorXd a_A, a_E;
	a_A = GramA.fullPivLu().solve(A);
	//a_E = GramE.fullPivLu().solve(E);
	//d.push_back(a_E.dot(E) - 1 / EigVal[0]);
	//d.push_back(a_A.dot(A) - 1 / EigVal[1]);
	return a_A.dot(A) / a_A.dot(a_A) - 1 / EigVal[1];
}

static dbl Equality3(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcObj(coef);
	dbl ret =  ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_omgEta);
	obj_L.terminate();
	return ret;
}

static dbl Equality4(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcObj(coef);
	dbl ret = ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Pos);
	obj_L.terminate();
	return ret;
}

static dbl Equality5(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcObj(coef);
	dbl ret = ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Zeta);
	obj_L.terminate();
	return ret;
}

static dbl Equality6(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcObj(coef);
	dbl ret = obj_LL.POS[NDIV - 1](0) - obj_L.POS[NDIV - 1](0);
	obj_L.terminate();
	return ret;
}

static dbl Equality7(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcObj(coef);
	dbl ret = obj_LL.POS[NDIV - 1](1) - obj_L.POS[NDIV - 1](1);
	obj_L.terminate();
	return ret;
}

static dbl Equality8(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
	VectorXd coef;
	vector<dbl> tmp = x;
	coef = Eigen::Map<Eigen::VectorXd>(&tmp[0], tmp.size());
	initializeForCalcObj(coef);
	dbl ret = obj_LL.POS[NDIV - 1](2) - obj_L.POS[NDIV - 1](2);
	obj_L.terminate();
	return ret;
}


int main(int argc, char** argv)
{
	initializing();
	memorizeGramMatrix();
	determineDimension();

	string DATA = "./data/" + file_name;
	if (_mkdir(DATA.c_str()) == 0.0) {
		cout << "dictionary " << DATA << " has been created\n";
	}
	else {
		int c, counter = 0;
		fprintf_s(stderr, "directory: %s already exists. \n", DATA.c_str());
		fprintf_s(stderr, "if you override the directory, input 'y'on keybord\n");
		fprintf_s(stderr, "if you don't, input any keys except for 'y' \n"); fflush(stderr);
		while ((c = getchar()) != '\n') {//バッファを吐き出すまで
			if (c == 'y' && counter == 0) {
				fprintf_s(stderr, "directory: %s has been overide\a\n", DATA.c_str());
				break;
			}
			else if (counter > 1000) {
				fprintf_s(stderr, "error at line:%d in func.\"%s()\"\n", __LINE__, __func__);
				exit(EXIT_FAILURE);
			}
			counter++;
		}
		if (counter != 0) {
			fprintf_s(stderr, "forced termination \n");
			exit(EXIT_FAILURE);
		}
	}
	opt OPTIMIZER(LN_SBPLX, NCOORD);
	//nlopt::opt local_opt(LN_SBPLX, NCOORD);
	nlopt::opt local_opt(LN_SBPLX, NCOORD);
	vector<dbl> EqEps(NCOND, 0.1);
	//OPTIMIZER.set_local_optimizer(local_opt);
	OPTIMIZER.set_min_objective(objectiveWrapper, NULL);
	//OPTIMIZER.add_equality_mconstraint(MultiCondFuncWrapper, NULL, EqEps);
	//OPTIMIZER.add_equality_constraint(Equality1, NULL, 0.001);
	//OPTIMIZER.add_equality_constraint(Equality2, NULL, 0.001);
	//OPTIMIZER.add_equality_constraint(Equality3, NULL, 0.001);
	//OPTIMIZER.add_equality_constraint(Equality4, NULL, 0.0001);
	//OPTIMIZER.add_equality_constraint(Equality5, NULL, 0.0001);
	//OPTIMIZER.add_equality_constraint(Equality6, NULL, 0.0001);
	//OPTIMIZER.add_equality_constraint(Equality7, NULL, 0.0001);
	//OPTIMIZER.add_equality_constraint(Equality8, NULL, 0.0001);
	vector<dbl> x0(NCOORD, 0.01);
	x0[NCOORD - 2] = 1.0;
	OPTIMIZER.set_ftol_rel(1.0e-4);
	//OPTIMIZER.set_stopval(1.0e-5);
	dbl f_opt;
	try {
		result ret = OPTIMIZER.optimize(x0, f_opt);
		cout << "optimize done\n";
		cout << ret;
		ofstream OBJ, COND, INEQ, COEF;
		OBJ.open("objfunc.txt");
		COND.open("conds.txt");
		INEQ.open("ineqs.txt");
		COEF.open("coef.txt");
		vector<dbl> c, i;
		vector<dbl> CONDS;
		coef = Eigen::Map<Eigen::VectorXd>(&x0[0], x0.size());
		CalcConds(NCOORD, coef, NCOND, CONDS);
		for (int p = 0; p < NCOND; p++) {
			//COND.write("%lf", c[p]);
			COND << CONDS[p];
			if (p != 6) COND << ",";
			cout << "i = " << p << "\n";
		}
		for (int i = 0; i < x0.size(); i++) {
			COEF << x0[i];
		}
		COND.close();
		OBJ << objective(NCOORD, coef);
		OBJ.close();
	}
	catch (exception& e) {
		std::cout << "nlopt failed: " << e.what() << std::endl;
	}
}