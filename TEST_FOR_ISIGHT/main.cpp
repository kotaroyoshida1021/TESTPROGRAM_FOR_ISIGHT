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
//Determine the dimension of coefficient for optimizing
static void determineDimension() {
	FullPivLU<MatrixXd> lu_decompA(GramA), lu_decompE(GramE), lu_decompD(GramD);
	cout << "rank = " << lu_decompA.rank() << ", " << lu_decompE.rank() << ", " << lu_decompD.rank() << "...";
	RankA = lu_decompA.rank();
	RankE = lu_decompE.rank();
	RankD = lu_decompD.rank();
	FILE* fp;
	errno_t err = fopen_s(&fp, "EIGEN_VALUES.txt", "r");
	vector<dbl> EigVal;
	dbl tmp;
	while (fscanf_s(fp, "%le\n", &tmp) != EOF) {
		EigVal.push_back(tmp);
	}
	if (RankE != Kdim) {// IF NULL SPACE EXISTS
		KerE = lu_decompE.kernel();
		NCOORD_ETA = Kdim - RankE;
	}
	else {
		MatrixXd EigE = EigVal[0] * MatrixXd::Identity(Kdim,Kdim) - GramE;
		FullPivLU<MatrixXd> lu_decompEigE(EigE);
		KerE = lu_decompEigE.kernel();
		NCOORD_ETA = Kdim - lu_decompEigE.rank();
	}
	if (RankD != Kdim) {// IF NULL SPACE EXISTS
		KerD = lu_decompD.kernel();
		NCOORD_DIST = Kdim - RankD;
	}
	else {
		MatrixXd Eig = EigVal[1] * MatrixXd::Identity(Kdim, Kdim) - GramD;
		FullPivLU<MatrixXd> lu_decompEig(Eig);
		KerD = lu_decompEig.kernel();
		NCOORD_DIST = Kdim - lu_decompEig.rank();
	}
	if (RankA != Kdim) {// IF NULL SPACE EXISTS
		KerA = lu_decompA.kernel();
		NCOORD_ALPHA = Kdim - RankA;
	}
	else {
		MatrixXd Eig = EigVal[2] * MatrixXd::Identity(Kdim, Kdim) - GramA;
		FullPivLU<MatrixXd> lu_decompEig(Eig);
		KerA = lu_decompEig.kernel();
		NCOORD_ALPHA = Kdim - lu_decompEig.rank();
	}
	PARAM_E = VectorXd::Zero(NCOORD_ETA);
	PARAM_D = VectorXd::Zero(NCOORD_DIST);
	PARAM_A = VectorXd::Zero(NCOORD_ALPHA);
	cout << "KerSize->" << KerA.size() / Kdim << ", " << KerE.size() / Kdim << ", " << KerD.size() / Kdim << "...";
	NCOORD = NCOORD_ALPHA + NCOORD_ETA + NCOORD_DIST + 2;
	cout << "NCOORD = " << NCOORD << "\n";
}
static dbl KroneckerDelta(dbl x, dbl y) {
	if (x == y) return 1.0;
	else return 0.0;
}
static dbl RadiusBasisFunc(dbl s_i, dbl s_j, VectorXd Params) { return Params(0) * exp(-0.5 * Square(s_i - s_j) / (Params(1) * Params(1))); }
static dbl RadiusBasisFuncSdot(dbl s, dbl s_j, VectorXd Params) { return -Params(0) * (s - s_j) * exp(-0.5 * Square(s - s_j) / Params(1) * Params(1)) / Square(Params(1))* Params(1); }
/*
	!! MUST USE IN FUNCTIONS WHICH ARE ARGUMENT OF NLOPT !!
	USAGE : DIVIDE INPUT VECTORS TO OPTIMIZIED FUNCTION
*/
static void divideCoef(VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << a.size();
#endif
	for (int i = 0; i < NCOORD - 2; i++) {
		if (i < NCOORD_ETA) {
			PARAM_E(i) = a[i];
		}
		else if ((i >= NCOORD_ETA) && (i < NCOORD_ETA + NCOORD_DIST)) {
			PARAM_D(i-NCOORD_ETA) = a[i];
		}
		else {
			PARAM_A(i - NCOORD_ETA - NCOORD_DIST) = a[i];
		}
	}
	Xi0Vec(0) = a[NCOORD-2];
	Xi0Vec(1) = a[NCOORD-1];
	//HILBERT COEFFICIENT

	OMG_ETA_H = KerE * PARAM_E;
	DIST_H = KerD * PARAM_D;
	BETA_H = KerA * PARAM_A;
	if (OMG_ETA_H.size() != Kdim) {
		cout << "error in func : " << __func__ << "\n";
		cout << "omegaEta size not match" << "\n";
		exit(1);
	}else if (DIST_H.size() != Kdim) {
		cout << "error in func : " << __func__ << "\n";
		cout << "Dist size not match" << "\n";
		exit(1);
	}
	else if (BETA_H.size() != Kdim) {
		cout << "error in func : " << __func__ << "\n";
		cout << "Beta size not match" << "\n";
		exit(1);
	}
}
/*
	THESE WERE FUNCTIONS FOR LINAER INTERPOLATE, EXPRESSED BY REPRODUSING KERNEL HILBERT SPACE 
*/
static dbl Beta(dbl s) {
	dbl tmp = 0.0;
	for (int i = 0; i < Kdim; i++) {
		tmp += BETA_H[i] * RadiusBasisFunc(s, i * Ds, AlphaParams);
	}
	return tan(tmp);
}

static dbl omgEta(dbl s) {
	dbl tmp = 0.0;
	for (int i = 0; i < Kdim; i++) {
		tmp += OMG_ETA_H(i) * RadiusBasisFunc(s, i * Ds, OmgEtaParams);
	}
	return tmp;
}

static dbl Dist(dbl s) {
	if (s == 0.0 || s == length_LL) {
		return 0.0;
	}
	else {
		dbl tmp = 0.0;
		for (int i = 0; i < Kdim; i++) {
			tmp += DIST_H(i) * RadiusBasisFunc(s, i * Ds, DistParams);
		}
		return tmp;
	}
}
/*
	THESE WERE FUNCTIONS FOR OBJECTIVE AND CONDITIONS
*/
static dbl omegaEtaWrap(dbl s) { return omegaEta(s); }
static dbl omegaXi(dbl s) { return DiffSqrt(kappa(s), omegaEta(s)); }
static dbl omegaZeta(dbl s) { return -omegaXi(s) * Beta(s); }


static void MemorizeFunctions() {
	for (int i = 0; i < NDIV; i++) {
		OMG_ETA[i] = omgEta(i * Ds);
		BETA[i] = Beta(i * Ds);
		DIST[i] = Dist(i * Ds);
	}
	omegaEta.set_Info(Ds, OMG_ETA);
	beta.set_Info(Ds, BETA);
	dist.set_Info(Ds, DIST);
}
static Coordinates obj_L(NDIV, length_LL, omegaXi,omegaEtaWrap,omegaZeta);
//FOR CALCULATE BENDING ENERGY
static dbl alphaSdot(dbl s) {
	dbl tmp = 0.0;
	for (int i = 0; i < Kdim; i++) {
		tmp += BETA_H(i) * RadiusBasisFuncSdot(s, i * Ds, AlphaParams);
	}
	return tmp;
}

static dbl objective_integrand(dbl s) {
#ifdef MY_DEBUG_MODE
	cout << "Now..." << __func__;
	cout << "...s->" << s << "\n";
#endif
	dbl TMP_1 = Square(omegaXi(s)) * (1 + Square(beta(s))) / (alphaSdot(s) + omegaEta(s));
	dbl TMP_2 = cos(atan(beta(s))) / (cos(atan(beta(s))) - Dist(s) * (alphaSdot(s) + omegaEta(s)));
	if (isnan(TMP_1) || isnan(TMP_2)) {
		cout << "isNaN is detected ->";
		cout << beta(s) << ", " << omegaEta(s) << ", " << Dist(s) << ", " << alphaSdot(s) << "\n";
		exit(1);
	}
	return fabs(TMP_1 * log(fabs(TMP_2)));
}

void initializeForCalcObj(VectorXd a) {
	divideCoef(a);
	MemorizeFunctions();
}

static dbl objective(int n, VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << "calculate objective...";
#endif
	initializeForCalcObj(a);
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
	dbl ret = ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, bind(&objective_integrand, _1));
	obj_L.terminate();
	return ret;
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
static dbl IntegrandForCond_omgEta(dbl s) {
	Vector3d zetaSdot = -omegaXi_LL(s) * obj_LL.eta(s) + omegaEta_LL(s) * obj_LL.xi(s);
	return Square((zetaSdot.dot(obj_L.xi(s)) - omegaEta(s)) / kappa(s));
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
	d.push_back(ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Pos));
	d.push_back(ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Zeta));
	d.push_back(ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_omgEta));
	for (int i = 0; i < 3; i++) {
		if (isnan(d[i])) {
			cout << "isNaN detected:-> I = " << i << "\n";
			exit(1);
		}
	}
	//cout << "all done\n";
	//COND = Eigen::Map<Eigen::VectorXd>(&d[0], d.size());
	COND = d;
	//COND *= 5.0;
	if (d.size() != NCOND) {
		cout << "error in func :" << __func__ << " -> quantity of condition is not match" << "\n";
		exit(1);
	}
#ifdef MY_DEBUG_MODE
	cout << "done:" << __func__ << "\n";
#endif
	vector<dbl>().swap(d);
	finalizeForCalcConds();
}

static dbl DistMax(dbl s) { return fabs(cos(atan(beta(s))) / (alphaSdot(s) + omgEta(s))); }
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
	for (int i = 0; i < NDIV; i++) {
		I.push_back(DIST[i] - DistMax(i * Ds));
		I.push_back(-DIST[i]);
	}
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
	//for (i = 0; i < NDIV; i++) {
	//	temp << obj_LL.POS[i](2) << " " << obj_LL.POS[i](0) << " " << obj_LL.POS[i](1) << "\n";
	//}
	temp << "\n";
	for (i = 0; i < NDIV; i++) {
		//temp << obj_L.POS[i](2) << " " << obj_L.POS[i](0) << " " << obj_L.POS[i](1) << "\n";
		temp << i * Ds << " " << Beta(i * Ds) << " " << omgEta(i * Ds) << "\n";
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
	CalcConds(NCOORD, coef, NCOND, COND);
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
		cout << "EVAL_COUNTER = " << EVAL_COUNTER << " f = " << ret << ", ";
		//string command = "splot \"" + F + "\" u 1:2:3 w lp pt 2 t \"k = " + ss.str() + "\n";
		gp.Command("set y2tics");
		string command = "plot \"" + F + "\" u 1:2 w l axis x1y1 t \"k = " + ss.str() +" Beta\", \"" + F + "\" u 1:3 w l axis x1y2 t \"k = " + ss.str() + " omgEta \n";
		gp.Command(command.c_str());
		if (EVAL_COUNTER % 100 == 0) {
			vector<dbl> CON;
			CalcConds(NCOND, coef, NCOND, CON);
			for (int i = 0; i < CON.size(); i++) {
				cout << " C[" << i << "] = " << CON[i] << ", ";
			}
			cout << "\n";
		}
	}
	++EVAL_COUNTER;
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
	opt OPTIMIZER(LN_AUGLAG, NCOORD);
	nlopt::opt local_opt(LN_SBPLX, NCOORD);
	vector<dbl> EqEps(NCOND, 0.0001);
	vector<dbl> IneqEps(NINEQ, 0.001);
	OPTIMIZER.set_local_optimizer(local_opt);
	OPTIMIZER.set_min_objective(objectiveWrapper, NULL);
	OPTIMIZER.add_inequality_mconstraint(MultiIneqFuncWrapper, NULL, IneqEps);
	OPTIMIZER.add_equality_mconstraint(MultiCondFuncWrapper, NULL, EqEps);
	vector<dbl> x0(NCOORD, 1.0e9);
	OPTIMIZER.set_ftol_rel(1.0e-5);
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