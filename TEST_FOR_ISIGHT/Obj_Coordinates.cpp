#include "pch.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <functional>
#include <fstream>
#include "GaussIntegral.h"
#include "Obj_Coordinates.h"
#include "LinearFunction.h"


using namespace std;
using namespace Eigen;
using namespace std::placeholders;

Coordinates::Coordinates(int N,dbl Length, dbl (*omegaXi)(dbl s), dbl (*omegaEta)(dbl s), dbl (*omegaZeta)(dbl s)) {
	n = N;
	Length = Length;
	Ds = Length / (dbl)(n-1);
	this->omegaXi = omegaXi;
	Coordinates::omegaEta = omegaEta;
	Coordinates::omegaZeta = omegaZeta;
}

Coordinates::~Coordinates() {
		vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(XI);
		vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(ETA);
		vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(ZETA);
		vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(POS);
}

void Coordinates::renew_omega(dbl(*omegaXi)(dbl s), dbl(*omegaEta)(dbl s), dbl(*omegaZeta)(dbl s)) {
	this->omegaXi = omegaXi;
	this->omegaEta = omegaEta;
	this->omegaZeta = omegaZeta;
}
void Coordinates::DetermineAxies(Vector3d& initXI, Vector3d& initETA, Vector3d& initZETA) {
	Vector3d k1_xi, k2_xi, k3_xi, k4_xi;
	Vector3d k1_eta, k2_eta, k3_eta, k4_eta;
	Vector3d k1_zeta, k2_zeta, k3_zeta, k4_zeta;
	Vector3d tmp;
	dbl S,DS;
	DS = Length / (dbl)(n - 1);
	InitializeVetctor();
	//cout << "start\n";
	XI.push_back(initXI); ETA.push_back(initETA); ZETA.push_back(initZETA);
	for (int i = 0; i < n - 1; i++) {
		//cout << "count = " << i << "\n";
		S = i * Ds;
		k1_xi = omegaZeta(S)*ETA[i] - omegaEta(S)*ZETA[i];
		k1_eta = -omegaZeta(S)*XI[i] + omegaXi(S)*ZETA[i];
		k1_zeta = omegaEta(S)*XI[i] - omegaXi(S)*ETA[i];
		//if (i == n - 2) fprintf(stderr, "koko?");
		tmp = XI[i] + k1_xi * (Ds / 2.0);
		XI.push_back(tmp);
		tmp = ETA[i] + k1_eta * (Ds / 2.0);
		ETA.push_back(tmp);
		tmp = ZETA[i] + k1_zeta * (Ds / 2.0);
		ZETA.push_back(tmp);

		S += Ds / 2.0;

		k2_xi = omegaZeta(S)*ETA[i+1] - omegaEta(S)*ZETA[i+1];
		k2_eta = -omegaZeta(S)*XI[i+1] + omegaXi(S)*ZETA[i+1];
		k2_zeta = omegaEta(S)*XI[i+1] - omegaXi(S)*ETA[i+1];
		//if (i == n - 2) fprintf(stderr, "koko?");
		XI[i + 1] = XI[i] + k2_xi * (Ds / 2.0);
		ETA[i + 1] = ETA[i] + k2_eta * (Ds / 2.0);
		ZETA[i + 1] = ZETA[i] + k2_zeta * (Ds / 2.0);

		S = S;
		//if (i == n - 2) fprintf(stderr, "koko?");
		k3_xi = omegaZeta(S)*ETA[i + 1] - omegaEta(S)*ZETA[i + 1];
		k3_eta = -omegaZeta(S)*XI[i + 1] + omegaXi(S)*ZETA[i + 1];
		k3_zeta = omegaEta(S)*XI[i + 1] - omegaXi(S)*ETA[i + 1];
		//if (i == n - 2) fprintf(stderr, "koko?");
		XI[i + 1] = XI[i] + k3_xi * (Ds);
		ETA[i + 1] = ETA[i] + k3_eta * (Ds);
		ZETA[i + 1] = ZETA[i] + k3_zeta * (Ds);

		S += Ds / 2.0;
		//if (i == n - 2) fprintf(stderr, "koko?");
		k4_xi = omegaZeta(S)*ETA[i + 1] - omegaEta(S)*ZETA[i + 1];
		k4_eta = -omegaZeta(S)*XI[i + 1] + omegaXi(S)*ZETA[i + 1];
		k4_zeta = omegaEta(S)*XI[i + 1] - omegaXi(S)*ETA[i + 1];
		//if (i == n - 2) fprintf(stderr, "koko?");
		XI[i + 1] = XI[i] + (k1_xi + 2.0*k2_xi + 2.0*k3_xi + k4_xi) * (Ds / 6.0);
		ETA[i + 1] = ETA[i] + (k1_eta + 2.0*k2_eta + 2.0*k3_eta + k4_eta) * (Ds / 6.0);
		ZETA[i + 1] = ZETA[i] + (k1_zeta + 2.0*k2_zeta + 2.0*k3_zeta + k4_zeta) * (Ds / 6.0);
		//if (i == n - 2) fprintf(stderr, "koko?");
	}
	//cout << "done\n";
	tmp = VectorXd::Zero(3);
	XI.push_back(tmp); ETA.push_back(tmp); ZETA.push_back(tmp);
	xi.set_Info(Ds,XI); eta.set_Info(Ds,ETA); zeta.set_Info(Ds,ZETA);
	GaussIntegral<Vector3d> G_I;
	G_I.SetGaussIntegralParams(GLI_15);
	//cout << "start2...";
	for (int i = 0; i <n; i++) {
		S = i * Ds;
		tmp = G_I.GaussIntegralFunc(0.0, S, bind(&VectorFunction::integrand, this->zeta, _1));
		POS.push_back(tmp);
	}
	//cout << "done\n";
	pos.set_Info(Ds,POS);
}

void Coordinates::fprint_shape(string filename) {
	ofstream writing;
	writing.open(filename, std::ios::out);
	ofstream dbg;
	dbl s = 0.0;
	dbg.open("Debug.txt", std::ios::out);
	for (int i = 0; i < n; i++) {
		s = i * Ds;
		writing << pos(s)(2) << " " << pos(s)(0) << " " << pos(s)(1) << "\n";
		dbg << omegaXi(i*Ds) << " " << omegaEta(i*Ds) << " " <<omegaZeta(i*Ds) << "\n";
	}
	dbg << "\n\n";
}

void Coordinates::terminate() {
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(XI);
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(ETA);
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(ZETA);
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>().swap(POS);
}



dbl Coordinates::omg_Xi(dbl s) {
	return omegaXi(s);
}

dbl Coordinates::omg_Eta(dbl s) {
	return omegaEta(s);
}

dbl Coordinates::omg_Zeta(dbl s) {
	return omegaZeta(s);
}

void Coordinates2D::DetermineAxies2D(int NDIV) {
	int n_div = NDIV;
	GaussIntegral<Vector2d> G_I2d;
	G_I2d.SetGaussIntegralParams(GLI_30);
	GaussIntegral<dbl> ScalarIntegralFunc;
	ScalarIntegralFunc.SetGaussIntegralParams(GLI_30);
	Vector2d tmp;
	for (int i = 0; i < n_div; i++) {
		THETA.push_back(ScalarIntegralFunc.GaussIntegralFunc(0.0, i * Ds, bind(&ScalarFunction::integrand, &LAMBDA, _1)));
		tmp << cos(THETA[i]), sin(THETA[i]);
		ZETA.push_back(tmp);
		tmp << -sin(THETA[i]), cos(THETA[i]);
		XI.push_back(tmp);
	}
	zeta.set_Info(Ds,ZETA);
}

void Coordinates::InitializeVetctor() {
	terminate();
}