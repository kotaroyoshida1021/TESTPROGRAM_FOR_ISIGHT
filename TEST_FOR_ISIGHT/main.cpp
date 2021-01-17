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
#define atanSigmoid(s) ((atan(s)*M_2_PI))

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
	VecIntergalFunc.SetGaussIntegralParams(GLI_15);
//SET GAUSSIAN INTEGRAL PARAMETERS OF VECTOR
	ScalarIntegralFunc.SetGaussIntegralParams(GLI_15);
	cout << "done\n";
//READ HYPERPARAMETERS FROM INPUT FILE
	FILE* fp;
	cout << "Initializing HyperParameters...";
	string fname = "HyperParams6.txt";
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
	NCOORD_ALPHA = vbase;
	NCOORD_ETA = vbase;
	NCOORD_DIST = 0;
	//NCOORD = NCOORD_ALPHA + NCOORD_ETA + NCOORD_DIST + 2;
	NCOORD = NCOORD_PER_PHI + NCOORD_PER_THETA + NCOORD_PER_UFUNC + 5;
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
	//NCOORD = NCOORD_ALPHA + NCOORD_ETA + NCOORD_DIST + 2;
	NCOORD = NCOORD_PER_PHI + NCOORD_PER_THETA + NCOORD_PER_UFUNC + 5;
	cout << "NCOORD = " << NCOORD << "\n";
}

static dbl RadiusBasisFunc(dbl s_i, dbl s_j, VectorXd Params) { return Params(0) * exp(-0.5 * Square(s_i - s_j) / (Params(1) * Params(1))); }
static dbl RadiusBasisFuncSdot(dbl s, dbl s_j, VectorXd Params) { return -Params(0) * (s - s_j) * exp(-0.5 * Square(s - s_j) / Params(1) * Params(1)) / Square(Params(1))* Params(1); }

/*
	DEFINE ZETA AND POS
*/
static dbl ufuncSdot(dbl s) { return exp(uSdot(s)); }
static dbl omgZeta_U(dbl s) { return 0.0; }
static dbl omgXi_U_Wrapper(dbl s) { return ufuncSdot(s)*omgXi_U(s); }
static dbl omgEta_U_Wrapper(dbl s) { return ufuncSdot(s)*omgEta_U(s);}
static Coordinates obj_UW(NDIV, length_LL, omgXi_U_Wrapper, omgEta_U_Wrapper, omgZeta_U);


static dbl ufunc(dbl s) { return ScalarIntegralFunc.GaussIntegralFunc(0.0, s, ufuncSdot); }

static Vector3d zeta_U_integrand(dbl s) { return ufuncSdot(s) * obj_UW.zeta(s);}

static Vector3d pos_U(dbl s) {
	return VecIntergalFunc.GaussIntegralFunc(0.0, s, zeta_U_integrand);
}
/*
	MEMORIZE ALPHA,OMG_ETA,DIST-> CALC FROM pos_U
*/
static void CalcAndMemorizeInformationOfDevelopableSurface() {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ ;
#endif
	obj_UW.DetermineAxies(xi0, eta0, zeta0);
	Vector3d zetaSdot = Vector3d::Zero(), GENE = Vector3d::Zero(), XI = Vector3d::Zero();
	dbl s = 0.;
	for (int i = 0; i < NDIV; i++) obj_UW.POS[i] = pos_U(i * Ds);
	obj_UW.pos.set_Info(Ds, obj_UW.POS);
	zetaSdot = -omegaXi_LL(s) * obj_LL.eta(s) + omegaEta_LL(s) * obj_LL.xi(s);
	Vector3d tmp = obj_LL.zeta(s).cross(obj_UW.zeta(s));
	Vector3d tmpXI = -obj_LL.zeta(s).cross(tmp.normalized());
	OMG_ETA[0] = zetaSdot.dot(tmpXI); ALPHA[0] = 0.0; DIST[0] = 0.0; DEV_CONDS[0] = 0.0;
	for (int I = 1; I < NDIV - 1; I++) {
		s = I * Ds;
		zetaSdot = -omegaXi_LL(s) * obj_LL.eta(s) + omegaEta_LL(s) * obj_LL.xi(s);
		GENE = obj_UW.pos(s) - obj_LL.pos(s);
		DIST[I] = GENE.norm();
		DEV_CONDS[I] = fabs(obj_LL.zeta(s).cross(obj_UW.zeta(s)).dot(GENE.normalized()));
		ALPHA[I] = asin(-GENE.normalized().dot(obj_LL.zeta(s)));
		//XI = GENE.normalized() - GENE.normalized().dot(obj_LL.zeta(s)) * obj_LL.zeta(s);
		XI = (GENE.normalized() / cos(ALPHA[I]) + obj_LL.zeta(s) * tan(ALPHA[I]));
		OMG_ETA[I] = zetaSdot.dot(XI);
	}
	//OMG_ETA[0] = OMG_ETA[1];
	GENE = obj_UW.pos(length_LL) - obj_LL.pos(length_LL);
	OMG_ETA[NDIV - 1] = OMG_ETA[NDIV - 2]; ALPHA[NDIV - 1] = ALPHA[NDIV - 2]; DIST[NDIV - 1] = GENE.norm(); DEV_CONDS[NDIV - 1] = fabs(obj_LL.zeta(length_LL).cross(obj_UW.zeta(length_LL)).dot(GENE.normalized()));
#ifdef MY_DEBUG_MODE
	cout << "done\n" ;
#endif
}


/*
	!! MUST USE IN FUNCTIONS WHICH ARE ARGUMENT OF NLOPT !!
	USAGE : DIVIDE INPUT VECTORS TO OPTIMIZIED FUNCTION
*/
static void divideCoef(VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << a.size();
#endif
	for (int i = 0; i < NCOORD-5; i++) {
		if (i < NCOORD_PER_PHI) {
			omgXi_U.a(i) = a(i);
		}
		else if (i >= NCOORD_PER_PHI && i < NCOORD_PER_PHI + NCOORD_PER_THETA) {
			omgEta_U.a(i - NCOORD_PER_PHI) = a(i);
		}
		else {
			uSdot.a(i - (NCOORD_PER_PHI + NCOORD_PER_THETA)) = a(i);
		}
	}
	//cout << "\n";
	Vector3d tmpXi, tmpEta;
	tmpXi << a[NCOORD - 5], a[NCOORD - 4], a[NCOORD - 3];
	xi0 = tmpXi.normalized();
	if (xi0(2) != 0.0) {
		tmpEta << a[NCOORD - 2], a[NCOORD - 1], -(a[NCOORD - 2] * xi0(0) + a[NCOORD - 1] * xi0(1)) / xi0(2);
	}
	else {
		tmpEta << -(a[NCOORD - 2] * xi0(1) + a[NCOORD - 1] * xi0(2)) / xi0(0), a[NCOORD - 2], a[NCOORD - 1];
	}
	eta0 = tmpEta.normalized();
	zeta0 = xi0.cross(eta0);
}	
/*
	THESE WERE FUNCTIONS FOR LINAER INTERPOLATE, EXPRESSED BY REPRODUSING KERNEL HILBERT SPACE 
*/

/*
	THESE WERE FUNCTIONS FOR OBJECTIVE AND CONDITIONS
*/
static dbl Volume();
static dbl Size();

static inline void MemorizeFunctions() {
	CalcAndMemorizeInformationOfDevelopableSurface();
	omegaEta.set_Info(Ds, OMG_ETA);
	alpha.set_Info(Ds, ALPHA);
	dist.set_Info(Ds, DIST);
	dev_conds.set_Info(Ds, DEV_CONDS);
	Vol = Volume();
	Si = Size();
}

void initializeForCalcObj(VectorXd a) {
	divideCoef(a);
	MemorizeFunctions();
}

static dbl objective(int n, VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << "calculate objective...";
#endif
	//initializeForCalcObj(a);
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
	initializeForCalcObj(a);
	dbl ret = ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, bind(&ScalarFunction::integrand, dev_conds, _1));//+ 5.0e-4 / Square(Volume());
	obj_UW.terminate();
	return ret;
}

static inline void initializeForCalcConds(VectorXd a) {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "...";
#endif
	divideCoef(a);
	MemorizeFunctions();
#ifdef MY_DEBUG_MODE
	cout << "done\n";
#endif
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

static void CalcConds(int n, VectorXd& a, int ncond, vector<dbl>& COND) {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "...";
#endif
	initializeForCalcConds(a);
	vector<dbl> d;
	VectorXd A, E;
	A = Eigen::Map<Eigen::VectorXd>(&ALPHA[0] + 1, ALPHA.size() - 2);
	E = Eigen::Map<Eigen::VectorXd>(&OMG_ETA[0] + 1, OMG_ETA.size() - 2);
	VectorXd D = Eigen::Map<Eigen::VectorXd>(&DIST[0] + 1, DIST.size() - 2);
	VectorXd a_A, a_E, a_D;
	a_A = GramA.fullPivLu().solve(A);
	a_E = GramE.fullPivLu().solve(E);
	a_D = GramD.fullPivLu().solve(D);
	//d.push_back((a_E.dot(E)) / (a_E.norm() * a_E.norm())-OmgEtaParams(2));
	//d.push_back((a_A.dot(A))/ (a_A.norm() * a_A.norm()) - AlphaParams(2));
	//d.push_back((a_D.dot(D)) / (a_D.norm() * a_D.norm()) - DistParams(2));
	d.push_back(1.0 * ((a_E.dot(E)) / (a_E.norm() * a_E.norm() * OmgEtaParams(2)) - 1));
	d.push_back(1.0 * ((a_A.dot(A)) / (a_A.norm() * a_A.norm() * AlphaParams(2)) - 1));
	d.push_back(1.0 * ((a_D.dot(D)) / (a_D.norm() * a_D.norm() * DistParams(2)) - 1));
	d.push_back(DIST[NDIV - 1]);
	for (int i = 0; i < 3; i++) d.push_back(1.0e+1 *(obj_UW.pos(length_LL)(i) - obj_LL.pos(length_LL)(i)));
	d.push_back(OMG_ETA[0] + 2.32193);
	//d.push_back(ALPHA[NDIV - 1] + 1.03664);
	//d.push_back(OMG_ETA[NDIV - 1] - 0.0541673);
	Vol = Volume();
	Si = Size();
	for (int i = 0; i < NCOND; i++) {
		if (isnan(d[i])) {
			cout << "isNaN detected:-> I = " << i << "\n";
			exit(1);
		}
	}
	COND = d;
	obj_UW.terminate();
	vector<dbl>().swap(d);
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
}
void initializeForCalcIneq(VectorXd a) {
	divideCoef(a);
	MemorizeFunctions();
}

static Vector3d GeneralPos(dbl s, dbl t) {
	return obj_LL.pos(s) + t * (obj_UW.pos(s) - obj_LL.pos(s));
}

static dbl VolumeIntegrand(dbl s, dbl t) {
	dbl Z = GeneralPos(s, t)(1);
	Vector3d Sdot, Tdot;
	Sdot = obj_LL.zeta(s) + t * (zeta_U_integrand(s) - obj_LL.zeta(s));
	Tdot = obj_UW.pos(s) - obj_LL.pos(s);
	
	return Z * sqrt(fabs(Sdot.dot(Sdot) * Tdot.dot(Tdot) - (Sdot.dot(Tdot)) * (Sdot.dot(Tdot))));
}

static dbl VolumeIntegrand2(dbl s) {
	return ScalarIntegralFunc.GaussIntegralFunc2Dmarginal2(s, 0.0, 1.0, bind(VolumeIntegrand, _1, _2));
}

static dbl Volume() {
	return ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, bind(VolumeIntegrand2, _1));
}

static dbl SizeIntegrand(dbl s) {
	dbl ret = (alpha.derivative(s) + omegaEta(s));
	if (ret < 0) {
		//return 0.5 * ret * Square(dist(s)) - cos(alpha(s)) * dist(s);
	}
	else {
		//return -0.5 * ret * Square(dist(s)) + cos(alpha(s)) * dist(s);
	}
	return fabs(-0.5 * ret * Square(dist(s)) + cos(alpha(s)) * dist(s));
}

static dbl Size() {
	return ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, bind(SizeIntegrand, _1));
}
static void calcIneqs(int n, VectorXd &a,int nineq,vector<dbl> &INEQ) {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "...";
#endif
	vector<dbl> I;
	initializeForCalcIneq(a);
	//INEQ = Eigen::Map<Eigen::VectorXd>(&I[0], I.size());
	//INEQ = I;
	for (int i = 0; i < NDIV; i++) {
		I.push_back(-pos_U(i * Ds)(1));
	}
	I.push_back(-Volume());
	//I.push_back(-Size());
	INEQ = I;
	if (I.size() != NINEQ) {
		cout << "error in func :" << __func__ << " -> quantity of condition is not match" << "\n";
		exit(1);
	}
	
	obj_UW.terminate();
	vector<dbl>().swap(I);
#ifdef MY_DEBUG_MODE
	cout << "all done\n";
#endif
}
//NLopt用WrapperSeries

void fprint_for_gnu(string FILE_NAME, VectorXd a) {
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
	Vector3d Gene = Vector3d::Zero();
	dbl al = 0.;
	for (i = 0; i < NDIV; i++) {
		//temp << obj_L.POS[i](2) + Gene(2) << " " << obj_L.POS[i](0) + Gene(0) << " " << obj_L.POS[i](1) + Gene(1) << "\n";
		temp << i * Ds << " " << ufunc(i * Ds) << " " << omegaEta(i * Ds) << " " << alpha(i * Ds) << " " << dist(i * Ds) << "\n";
	}
	temp << "\n\n";
	temp.close();
	obj_UW.terminate();
}

void fprint_for_Files(string FOLDER_NAME,vector<dbl> x) {
	cout << "Start..FprintFiles...";
	VectorXd COEF = Eigen::Map<Eigen::VectorXd>(&x[0], x.size());
	initializeForCalcObj(COEF);
	string shape_name = FOLDER_NAME + "/shape.txt";
	string func_name = FOLDER_NAME + "/functions.txt";
	string wire_name = FOLDER_NAME + "/wire.txt";
	ofstream shape, func, wire;
	shape.open(shape_name, std::ios::out);
	func.open(func_name, std::ios::out);
	wire.open(wire_name, std::ios::out);
	if (shape.fail() || func.fail() || wire.fail()) {
		cout << "cannot open file\n";
		exit(1);
	}
	for (int i = 0; i < NDIV; i++) {
		cout << i << ", ";
		wire << obj_LL.POS[i](2) << " " << obj_LL.POS[i](0) << " " << obj_LL.POS[i](1) << "\n";
		//shape << obj_L.POS[i](2) << " " << obj_L.POS[i](0) << " " << obj_L.POS[i](1) << "\n";
		shape << obj_UW.pos(i * Ds)(2) << " " << obj_UW.pos(i * Ds)(0) << " " << obj_UW.pos(i * Ds)(1) << "\n";
		//ここから
		func << i * Ds << " " << ALPHA[i] << " " << OMG_ETA[i] << " " << DIST[i] << " " << obj_UW.zeta(i * Ds)(2) << " " << obj_UW.zeta(i * Ds)(0) << " " << obj_UW.zeta(i * Ds)(1) << "\n";
	}
	shape.close();
	func.close();
	wire.close();
	obj_UW.terminate();
	//obj_L.terminate();
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
	vector<dbl>COND;
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
		FILE* fp;
		errno_t err = fopen_s(&fp, "EvalFunc_and_Conds.csv", "a");
		//fprintf_s(stderr, "\r%8d | %6.3e \r", EVAL_COUNTER, ret);
		//cout << "EVAL_COUNTER = " << scientific << setprecision(5) << EVAL_COUNTER << " f = " << scientific << setprecision(5) << ret << ", ";
		cout << scientific << setprecision(5) << EVAL_COUNTER << "  " << scientific << setprecision(5) << ret << "  ";
		//string command = "splot \"" + F + "\" u 1:2:3 w lp pt 2 t \"k = " + ss.str() + "\n";
		gp.Command("set y2tics");
		//gp.Command("set term qt 1");
		string command = "plot \"" + F + "\" u 1:2 w l axis x1y1 t \"k = " + ss.str() +" ufunc\", \"" + F + "\" u 1:3 w l axis x1y2 t \"k = " + ss.str() + " omgEta \n";
		gp.Command(command.c_str());
		gp_shape.Command("set y2tics");
		string command2 = "plot \"" + F + "\" u 1:4 w l axis x1y1 t \"k = " + ss.str() + " alpha\", \"" + F + "\" u 1:5 w l axis x1y2 t \"k = " + ss.str() + " dist \n";
		gp_shape.Command(command2.c_str());
		fprintf_s(fp, "%d,%lf,", EVAL_COUNTER, ret);
		if (EVAL_COUNTER % 100 == 0) {
			vector<dbl> CONDS;
			CalcConds(NCOORD, coef, NCOND, CONDS);
			for (int i = 0; i < NCOND; i++) {
				//cout << "COND[" << i << "]=" << scientific << setprecision(5) << CONDS[i] << ", ";
				cout << scientific << setprecision(5) << CONDS[i] << "  ";
				fprintf_s(fp, "%lf,", CONDS[i]);
			}
			fprintf_s(fp, "%lf,%lf,", Vol, Si);
			fprintf_s(fp, "\n");
			fclose(fp);
			cout << scientific << setprecision(5) << Vol << " " << scientific << setprecision(5) << Si;
			//cout << ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_omgEta) << ", " << ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Pos) << ", " << ScalarIntegralFunc.GaussIntegralFunc(0.0, length_LL, IntegrandForCond_Zeta);
			cout << "\n";
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

static dbl Equality9(const vector<dbl>& x, vector<dbl>& grad, void* my_func_data) {
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


int main(int argc, char** argv)
{
	initializing();
	memorizeGramMatrix();
	determineDimension();
	//DBG用初期化
	FILE* fp_P;
	errno_t err = fopen_s(&fp_P, "EvalFunc_and_Conds.csv", "w");
	fclose(fp_P);
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
	cout << "ITERATION    OBJECTIVE    ";
	if (NCOND != 0) {
		for(int i=0;i<NCOND;i++) cout << "COND[" << i << "]    ";
	}
	cout << "\n";
	opt OPTIMIZER(LN_AUGLAG, NCOORD);
	//nlopt::opt local_opt(LN_SBPLX, NCOORD);
	nlopt::opt local_opt(LN_SBPLX, NCOORD);
	vector<dbl> EqEps(NCOND, 0.0001);
	EqEps[0] = 0.0001;
	EqEps[1] = 0.0001;
	EqEps[2] = 0.0001;
	vector<dbl>IneqEps(NINEQ, 0.0001);
	OPTIMIZER.set_local_optimizer(local_opt);
	OPTIMIZER.set_min_objective(objectiveWrapper, NULL);
	OPTIMIZER.add_equality_mconstraint(MultiCondFuncWrapper, NULL, EqEps);
	OPTIMIZER.add_inequality_mconstraint(MultiIneqFuncWrapper, NULL, IneqEps);
	vector<dbl> x0(NCOORD, 0.0001);
	x0[NCOORD - 5] = 1.0;
	x0[NCOORD - 1] = 1.0;
	//for (int i = 0; i < NCOORD - 5; i++) {
	//	if (i < NCOORD_PER_PHI) {
	//		x0[i] = 0.00001 * (i);
	//	}
	//	else if (i >= NCOORD_PER_PHI && i < NCOORD_PER_PHI + NCOORD_PER_THETA) {
	//		x0[i] = 0.0001;
	//	}
	//	else {
	//		x0[i] = 0.001 * (i - (NCOORD_PER_PHI + NCOORD_PER_THETA) + 1);
	//	}
	//}
	//x0[NCOORD - 5] = 1.0; x0[NCOORD - 4] = 0.000001; x0[NCOORD - 3] = 0.0001;
	//x0[NCOORD - 2] = 0.000001; x0[NCOORD - 1] = 1.0;
	OPTIMIZER.set_ftol_rel(1.0e-4);
	OPTIMIZER.set_maxeval(200000000);
	//OPTIMIZER.set_stopval(1.0e-5);
	dbl f_opt;
	try {
		result ret = OPTIMIZER.optimize(x0, f_opt);
		cout << "optimize done\n";
		cout << ret;
		for (int J = 0; J < NCOORD; J++) {
			cout << x0[J] << ", ";
		}
		cout << "\n";
		fprint_for_Files(DATA, x0);
		ifstream is("EvalFunc_and_Conds.csv", ios::in | ios::binary);
		ofstream os(DATA + "/EvalFunc_and_Conds.csv", ios::out | ios::binary);
		os << is.rdbuf();
	}
	catch (exception& e) {
		std::cout << "nlopt failed: " << e.what() << std::endl;
	}
}