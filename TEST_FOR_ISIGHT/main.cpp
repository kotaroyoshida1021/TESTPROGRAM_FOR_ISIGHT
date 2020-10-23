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

static void initializing(string InputFname) {
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
		AlphaParams(i) = HyperParams(i);
		OmgEtaParams(i) = HyperParams(3 + i);
		DistParams(i) = HyperParams(3*2 + i);
	}
	fclose(fp);
//Inputファイルから係数を読み出す
	cout << "Reading coefficient from file...";
	err = fopen_s(&fp, InputFname.c_str(), "r");
	//vector<dbl> InputCoef;
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
	cout << "done\n";
//補間する対象は，出力値であるとする．
	AlphaCoef = VectorXd::Zero(dim);
	OmgEtaCoef = VectorXd::Zero(dim);
	DistCoef = VectorXd::Zero(dim);
//仮
	for (int i = 0; i < dim; i++) {
		AlphaCoef(i) = InputCoef[i];
		OmgEtaCoef(i) = InputCoef[i + dim];
		DistCoef(i) = InputCoef[i + 2 * dim];
	}
//初期化
	ALPHA = VectorXd::Zero(dim);
	OMG_ETA = VectorXd::Zero(dim);
	DIST = VectorXd::Zero(dim);

	Xi0Vec(0) = InputCoef[3 * dim];
	Xi0Vec(1) = InputCoef[3 * dim + 1];
	//coef->initializing
	S = VectorXd::Zero(dim);
	for (int i = 0; i < dim; i++) {
		S(i) = i * length_LL / (dim - 1);
	}
	cout << "all done\n";
}

static dbl RadiusBasisFunc(dbl s_i, dbl s_j, VectorXd Params) { return Params(0) * exp(-0.5 * Square(s_i - s_j) / (Params(1) * Params(1))); }
static dbl RadiusBasisFuncSdot(dbl s, dbl s_j, VectorXd Params) { return -Params(0) * (s - s_j) * exp(-0.5 * Square(s - s_j) / Params(1) * Params(1)) / Square(Params(1)); }

//点群をRBF補間するやつ => return sum_i=0^N k(s,s_i;Param)*Coef_i;
static dbl RBF_Interpolate(dbl s, VectorXd Param, VectorXd Coef) {
	VectorXd kernel = VectorXd::Zero(dim);
	for (int i = 0; i < dim; i++) {
		//kernel(i) = exp(-0.5 * (s - S(i)) * (s - S(i)) / (AlphaParams[1] * AlphaParams[1]));
		kernel(i) = RadiusBasisFunc(s, S(i), Param);
	}
	return Coef.dot(kernel);
}

static dbl alpha(dbl s) { return RBF_Interpolate(s,AlphaParams,ALPHA); }//出力をRBF補間
static dbl omgEta(dbl s) { return RBF_Interpolate(s, OmgEtaParams, OMG_ETA); }//出力をRBF補間
static dbl dist(dbl s) { return RBF_Interpolate(s, DistParams, DIST); }//出力をRBF補間

static dbl alphaSdot(dbl s) {
	VectorXd kernel = VectorXd::Zero(dim);
	for (int i = 0; i < dim; i++) {
		kernel(i) = RadiusBasisFuncSdot(s, S(i), AlphaParams);
		//kernel(i) = (s - S(i)) * exp(-0.5 * Square(s - S(i)) / AlphaParams(1) * AlphaParams(1)) / Square(AlphaParams(1));
	}
	return ALPHA.dot(kernel);
}
//zeta計算用
static dbl phi(dbl s) { return acos(omgEta(s) / kappa(s)); }
static dbl omgXi(dbl s) { return kappa(s) * sin(phi(s)); }
static dbl omgZeta(dbl s) { return -omgXi(s) * tan(alpha(s)); }

static Coordinates obj_L(NDIV, length_LL, omgXi, omgEta, omgZeta);
//パラメータを不等式制約を考慮できるように非線形変換する．

static void calcKernMatrix(VectorXd Params, MatrixXd& Mat) {
	Mat = MatrixXd::Zero(dim, dim);
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			Mat(i, j) = RadiusBasisFunc(i * Ds, j * Ds, Params);
		}
	}
	MatrixXd I = MatrixXd::Identity(dim, dim);
	Mat += Params(2) * I;

}

#define AtanSigmoid(s) (2.0*atan(s)/M_PI)
static dbl SingularDist(dbl s) {
	return fabs(cos(alpha(s)) / (alphaSdot(s) + omgEta(s)));
}

static void considerIneqs() {
	//出力は非線形変換された中身を返しているため，適切に変換してやる．
	AlphaCoef = VectorXd::Zero(dim);
	OmgEtaCoef = VectorXd::Zero(dim);
	DistCoef = VectorXd::Zero(dim);
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "\n";
#endif
	for (int i = 0; i < dim; i++) {
		AlphaCoef(i) = InputCoef[i];
		OmgEtaCoef(i) = kappa(S(i))*AtanSigmoid(InputCoef[i + dim]);
		
		//DistCoef(i) = InputCoef[i + 2 * dim];
	}
	MatrixXd KA, KE, KD;
	calcKernMatrix(AlphaParams, KA);
	calcKernMatrix(OmgEtaParams, KE);
	Eigen::FullPivLU <Eigen::MatrixXd> lu_A(KA), lu_E(KE);
	ALPHA = lu_A.solve(AlphaCoef);
	OMG_ETA = lu_E.solve(OmgEtaCoef);
	for (int i = 0; i < dim; i++) {
		DistCoef(i) = SingularDist(S(i)) * Square(sin(InputCoef[i + 2 * dim]));
	}
	calcKernMatrix(DistParams, KD);
	if (KD.determinant() == 0.0) {
		KD += 1.0e-8 * MatrixXd::Identity(dim, dim);
	}
	FullPivLU <Eigen::MatrixXd> lu_D(KD);
	DIST = lu_D.solve(DistCoef);
	for (int i = 0; i < dim; i++) {
		if (isnan(DistCoef(i))) {
			cout << "NaN is DistCoef >>";
			cout << "\n";
			exit(1);
		}
	}
	cout << DistCoef;
	cout << "\n";
	
	//制約を加える．
	
	//カーネルマトリクスの設定
	
	
}



//目的関数の被積分関数
static dbl objective_integrand(dbl s) {
	Vector3d zetaSdot = omegaEta_LL(s) * obj_LL.xi(s);
	if (isnan(Square(obj_LL.zeta(s).dot(obj_L.zeta(s)) - 1)) || isnan(Square(zetaSdot.dot(obj_L.xi(s)) - omgEta(s)))) {
		cout << "isNan is found! condition>>";
		cout << "param = " << s << ", zeta = " << obj_LL.zeta(s).dot(obj_L.zeta(s)) - 1 << ", OmgEta = " << zetaSdot.dot(obj_L.xi(s)) - omgEta(s) << "\n";
		exit(1);
	}
	return Square(obj_LL.zeta(s).dot(obj_L.zeta(s)) - 1) + Square(zetaSdot.dot(obj_L.xi(s))-omgEta(s));
}



static dbl objective(VectorXd &a) {
#ifdef MY_DEBUG_MODE
	cout << "calculate objective...";
#endif
	considerIneqs();
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
static MatrixXd PartialKernelMatrix(int NUM,VectorXd Params) {
	MatrixXd pK;
	pK = MatrixXd::Identity(Kdim, Kdim);
	if (NUM = !2) {
		for (int i = 1; i < NDIV - 1; i++) {
			for (int j = 1; j < NDIV - 1; j++) {
				switch (NUM) {
				case 0:
					pK(i, j) = RadiusBasisFunc(i * Ds, j * Ds, Params) / Params(0);
				case 1:
					pK(i, j) = -(Square(i * Ds - j * Ds)) / (Square(Params(1)) * Params(1)) * RadiusBasisFunc(i * Ds, j * Ds, Params);
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

static void calcGramMatrix(VectorXd Params, MatrixXd &Mat) {
	Mat = MatrixXd::Zero(Kdim, Kdim);
	for (int i = 1; i < NDIV - 1; i++) {
		for (int j = 1; j < NDIV - 1; j++) {
			Mat(i-1, j-1) = RadiusBasisFunc(i * Ds, j * Ds, Params);
		}
	}

}
static dbl DeterminantParamDerivative(int NUM, VectorXd Params) {
	MatrixXd K, Kinv, Kdot, TMP;
	calcGramMatrix(Params, K);
	K += Params(2) * MatrixXd::Identity(Kdim, Kdim);
	Kdot = PartialKernelMatrix(NUM, Params);
	Kinv = K.inverse();
	TMP = Kinv * Kdot;
	return K.determinant() * TMP.trace();

}

static MatrixXd ParticalKinverse(int NUM, VectorXd Params) {
	MatrixXd K, Kinv, Kdot;
	calcGramMatrix(Params, K);
	
	K += Params(2) * MatrixXd::Identity(Kdim, Kdim);
	Kdot = PartialKernelMatrix(NUM, Params);
	Kinv = K.inverse();
	return -Kinv * Kdot * Kinv;
}

static vector<dbl> CalcConds() {
#ifdef MY_DEBUG_MODE
	cout << "now..." << __func__ << "\n";
#endif
	considerIneqs();
	vector<dbl> d;
	dbl C[3];
	MatrixXd KhatA, KhatE, KhatD;
	MatrixXd I = MatrixXd::Identity(Kdim, Kdim);
	KhatA = MatrixXd::Identity(dim, Kdim);
	KhatE = MatrixXd::Identity(dim, Kdim);
	KhatD = MatrixXd::Identity(dim, Kdim);
	VectorXd AlphaCoef2 = VectorXd::Zero(dim - 2);
	VectorXd OmgEtaCoef2 = VectorXd::Zero(dim - 2);
	VectorXd DistCoef2 = VectorXd::Zero(dim - 2);
#ifdef MY_DEBUG_MODE
	cout << "Initialize Khats...";
#endif
/*
	for (int i = 0; i < dim-2; i++) {
		for (int j = 1; j < NDIV - 1; j++) {
			KhatA(i, j-1) = RadiusBasisFunc(S(i+1), j * Ds, AlphaParams);
			KhatE(i, j-1) = RadiusBasisFunc(S(i+1), j * Ds, OmgEtaParams);
			KhatD(i, j-1) = RadiusBasisFunc(S(i+1), j * Ds, DistParams);
		}
	}
	KhatA += AlphaParams(2) * I;
	KhatE += OmgEtaParams(2) * I;
	KhatD += DistParams(2) * I;
*/
#ifdef MY_DEBUG_MODE
	cout << "all done\n";
#endif
	MatrixXd KA, KE, KD;
#ifdef MY_DEBUG_MODE
	cout << "calculate GramMatrix...";
#endif
	for (int i = 0; i < dim - 2; i++) {
		AlphaCoef2(i) = ALPHA(i + 1);//AlphaCoef(i + 1);
		OmgEtaCoef2(i) = OMG_ETA(i + 1);//OmgEtaCoef(i + 1);
		DistCoef2(i) = DIST(i + 1);//DistCoef(i + 1);
	}
	calcGramMatrix(AlphaParams, KA);
	calcGramMatrix(OmgEtaParams, KE);
	calcGramMatrix(DistParams, KD);
	dbl DET[3];
	DET[0] = KA.determinant();
	DET[1] = KE.determinant();
	DET[2] = KD.determinant();
#ifdef MY_DEBUG_MODE
	cout << "all done\n";
	cout << "PushBacks and calculate PartialKernel....";
#endif

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (DET[j] == 0.0) {
				DET[j] = 1.0e-6;
			}
		}
		//C[0] = -0.5 * DeterminantParamDerivative(i, AlphaParams) / DET[0] - 0.5 * AlphaCoef.dot(KhatA * ParticalKinverse(i, AlphaParams) * KhatA.transpose() * AlphaCoef);
		//C[1] = -0.5 * DeterminantParamDerivative(i, OmgEtaParams) / DET[1] - 0.5 * OmgEtaCoef.dot(KhatE * ParticalKinverse(i, OmgEtaParams) * KhatE.transpose() * OmgEtaCoef);
		//C[2] = -0.5 * DeterminantParamDerivative(i, DistParams) / DET[2] - 0.5 * DistCoef.dot(KhatD * ParticalKinverse(i, DistParams) * KhatD.transpose() * DistCoef);
		C[0] = -0.5 * DeterminantParamDerivative(i, AlphaParams) / DET[0] + 0.5 * AlphaCoef2.dot(PartialKernelMatrix(i, AlphaParams) * AlphaCoef2);
		C[1] = -0.5 * DeterminantParamDerivative(i, OmgEtaParams) / DET[1] + 0.5 * OmgEtaCoef2.dot(PartialKernelMatrix(i, OmgEtaParams) * OmgEtaCoef2);
		C[2] = -0.5 * DeterminantParamDerivative(i, DistParams) / DET[2] + 0.5 * DistCoef2.dot(PartialKernelMatrix(i, DistParams) * DistCoef2);
		for (int p = 0; p < 3; p++) {
			d.push_back(C[p]);
			if (isnan(C[p])) {
				switch (p) {
				case 0:
					cout << "isNaN is alpha" << "\n";
					cout << DeterminantParamDerivative(i, AlphaParams) << ", "<<AlphaCoef2.dot(PartialKernelMatrix(i, AlphaParams) * AlphaCoef2) << "\n";
					cout << "Mat\n";
					cout << PartialKernelMatrix(i, AlphaParams);
					
					exit(1);
				case 1:
					cout << "isNaN is omgEta" << "\n";
					cout << DeterminantParamDerivative(i, OmgEtaParams) << "\n";
					cout << PartialKernelMatrix(i, OmgEtaParams);
					exit(1);
				case 2:
					cout << "isNaN is Dist" << "\n";
					cout << DeterminantParamDerivative(i, DistParams) << "\n";
					cout << PartialKernelMatrix(i, DistParams);
					cout << "coef\n";
					cout << DIST;
					exit(1);
				default:
					cout << "Wrong p num;p=" << p << "\n";
					exit(1);
				}

			}
		}
	}
#ifdef MY_DEBUG_MODE
	cout << "all done\n";
#endif
	d.push_back(DistCoef[0]);
	d.push_back(DistCoef(DistCoef.size() - 1));
	return d;
#ifdef MY_DEBUG_MODE
	cout << "all done\n";
#endif
}

static dbl DistMax(dbl s) {
	return cos(alpha(s)) / (alphaSdot(s) + omgEta(s));
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
#ifdef MY_DEBUG_MODE
	ofstream FILE;
	FILE.open("INPUT.txt");
	for (int i = 0; i < dim*3; i++) {
		//FILE.write("%lf ", i * 0.01);
		FILE << i * 0.0001;
		FILE << "\n";
	}
	FILE << 0.01 << "\n";
	FILE << 0.2;
	FILE.close();
#endif
	string INPUT_FILENAME;
	bool FILE_MATCH = TRUE;
	INPUT_FILENAME = string(argv[1]);
	if (INPUT_FILENAME != "INPUT.txt") {
		cout << "Wrong File Name: Check Input file name and argument" << "\n";
		exit(1);
	}
	initializing(INPUT_FILENAME);

	ofstream OBJ, COND, INEQ;
	OBJ.open("objfunc.txt");
	COND.open("conds.txt");
	INEQ.open("ineqs.txt");
	vector<dbl> c, i;
	c = CalcConds();
	i = calcIneqs();
	OBJ << objective(AlphaCoef);
	for (int p = 0; p < c.size(); p++) {
		//COND.write("%lf", c[p]);
		COND << c[p];
		if (p != c.size()) COND << ",";
	}
	for (int j = 0; j < i.size(); j++) {
		//INEQ.write("%lf", i[j]);
		INEQ << i[j];
		if (j != i.size()) INEQ << ",";
	}
	
}