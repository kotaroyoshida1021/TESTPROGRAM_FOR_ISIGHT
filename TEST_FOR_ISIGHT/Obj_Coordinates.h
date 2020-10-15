#pragma once
#include <Eigen/Core>
#include <vector>
#include <fstream>
#include "LinearFunction.h"
#include "GaussIntegral.h"

using namespace std;
using namespace Eigen;



class Coordinates{
private:
	int n; //ï™äÑêî
	dbl Length;
	dbl (*omegaXi)(dbl s);
	dbl (*omegaEta)(dbl s);
	dbl (*omegaZeta)(dbl s);
	void InitializeVetctor();
public:
	Coordinates(int n,dbl Length, dbl (*omegaXi)(dbl s),dbl (*omegaEta)(dbl s),dbl (*omegaZeta)(dbl s));
	~Coordinates();
	dbl Ds;
	//GaussIntegral<Vector3d> Integral;
	void renew_omega(dbl(*omegaXi)(dbl s), dbl(*omegaEta)(dbl s), dbl(*omegaZeta)(dbl s));
	void DetermineAxies(Vector3d& initXI, Vector3d& initETA,Vector3d& initZETA);
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> XI;
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> ETA;
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> ZETA;
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> POS;
	VectorFunction xi;
	VectorFunction eta;
	VectorFunction zeta;
	VectorFunction pos;
	dbl omg_Xi(dbl s);
	dbl omg_Eta(dbl s);
	dbl omg_Zeta(dbl s);
	void fprint_shape(string filename);
	void terminate();
};

class Coordinates2D {
private:
	int ndiv;
	dbl Length;
	dbl Ds;
	ScalarFunction LAMBDA;
public:
	void set_Info(dbl delS, ScalarFunction &LMD) {
		Ds = delS;
		LAMBDA = LMD;
	};
	vector<dbl> THETA;
	vector<Vector2d,aligned_allocator<Vector2d>> ZETA;
	vector<Vector2d, aligned_allocator<Vector2d>> XI;
	vector<Vector2d, aligned_allocator<Vector2d>> POS;
	ScalarFunction theta;
	VecFunc<Vector2d> zeta;
	VecFunc<Vector2d> xi;
	VecFunc<Vector2d> pos;
	void DetermineAxies2D(int NDIV);
};