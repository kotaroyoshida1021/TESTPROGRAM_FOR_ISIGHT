#pragma once
#pragma once
#include "pch.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>
#include "gnuplot.h"
#include "Nelder_Mead.h"

constexpr dbl PotentialWeight = 1.0;
class MultiSimp :public Nelder_Mead{
private:
	int nvar;
	dbl L_neld;
	dbl ALPHA, BETA,C;
	dbl ConditionVal;
	VectorXd optimizedVec;
	VectorXd mu;
	VectorXd lm;
	VectorXd s;
	VectorXd r;
	VectorXd CondsVal,IneqsVal;
	VectorXd gvalue;
	VectorXd hvalue;
	dbl length, epsn;
	int NCOND, NINEQ;
	int ITERATETION_NUM;
	int timeoutn;
	dbl(*objective)(int n, VectorXd &x);
	void(*calc_conds)(int n_var, VectorXd &x,int conds_num,VectorXd &CONDS);
	void(*calc_ineqs)(int n_var, VectorXd &x,int ineqs_num,VectorXd &INEQS);
	
public:
	MultiSimp(int n, dbl(*objectiveFunc)(int n, VectorXd &x), VectorXd& x, dbl length, int timeoutn, dbl epn, int ncond, void(*Conds)(int n, VectorXd &x, int ncond, VectorXd &G), int nineq, void(*Ineqs)(int n, VectorXd &x, int nineq, VectorXd& H), int IterateNum, dbl ConvergenceCondition, VectorXd& Lambda, VectorXd& Mu);
	dbl LagrangeFunc(int n, VectorXd &x);
	void Initialize();
	void Set_MultiSimpParams(int n, dbl(*objectiveFunc)(int n, VectorXd &x), int ncond, void(*Conds)(int n, VectorXd &x, int ncond, VectorXd &G), int nineq, void(*Ineqs)(int n, VectorXd &x, int nineq, VectorXd& H), VectorXd& x, dbl length, int IterateNum, dbl ConvergenceCondition, VectorXd& Lambda, VectorXd& Mu, int timeoutn, dbl epn);
	status Launch_MultiSimp(string filename,void(*printFunc)(string filename));
	~MultiSimp();
	
};