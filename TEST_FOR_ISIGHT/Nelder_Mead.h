#pragma once
#include "pch.h"
#include <Eigen/Core>
#include <vector>
#include "gnuplot.h"

using namespace std;
using namespace Eigen;

constexpr auto SKIPTIME = 100;	/* print interval for debugging */
constexpr auto MONITORTIME = 100;	/* for Gnuplot RealTimeDisplay */;
constexpr auto UPPERfvalue = 1.0e15;	/* threshold value of fvalue */
#define SetGPMonitor 0

struct ObjectiveFunction {
public:
	VectorXd a;
	dbl val;
public:

	ObjectiveFunction();

	ObjectiveFunction(VectorXd &a,dbl val):a(a),val(val){
		//fprintf_s(stderr, "Objective contractor\n");
	}
	
	~ObjectiveFunction() {
		//cout << "Launch_destructor\n";
	}

	
	static bool comparing(ObjectiveFunction &f1, ObjectiveFunction &f2) {
		//return (f1->val < f2->val);
		return f1.val < f2.val;
	}
	ObjectiveFunction(const ObjectiveFunction& OBJ) :a(OBJ.a), val(OBJ.val) {
	//	fprintf_s(stderr, "Objective copied\n");
	}

	//ObjectiveFunction(ObjectiveFunction&& OBJ) noexcept : a(OBJ.a), val(OBJ.val) { fprintf_s(stderr, "Call move constructor\n");}

	
};
class Nelder_Mead {
private:
	int nvar;
	int TIMEOUT;
	dbl alpha, beta, gamma, delta;
	//vector<ObjectiveFunction> simplex;
	vector<pair<dbl, VectorXd>> simplex;
	function<dbl(int, VectorXd&)> obj;
	vector<VectorXd, Eigen::aligned_allocator<VectorXd>> inits;
	VectorXd x_init,x_centroid,x_reflect,x_contract,x_expand,x_inside;
	dbl L;
	dbl EPS;
	void initializing_num();
	void SetAdaptiveParams(int n);
	void initialize();
	void SearchSimplex();
	void calc_Centroid();
	dbl calc_variance();
	void shrink();
	string filename;
	void(*printFunc)(string filename);
public:
	int n_expand, n_reflect, n_contract, n_shrink , n_inside;
	//Nelder_Mead(int n, function<dbl(int, VectorXd&)> func, VectorXd initvec, dbl length, int timeout,dbl eps);
	void Set_Nelder_Mead(int n, function<dbl(int, VectorXd&)> func, VectorXd &initvec, dbl length, int timeout, dbl eps);
	status launch_NelderMead(dbl& fopt,int Iterate_num);
	void fprintCoeffcients();
	void set_printParams(string FILENAME, void(*priFunc)(string filename)) {
		filename = FILENAME;
		printFunc = priFunc;
	};
	VectorXd return_vec() {
		return simplex[0].second;
	}
};