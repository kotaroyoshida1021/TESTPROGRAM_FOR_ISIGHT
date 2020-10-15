#include "pch.h"
#include <iostream>
#include <Eigen/Core>
#include <functional>
#include <vector>
#include <fstream>
#include "gnuplot.h"
#include "MultiSimp.h"
#define _USE_MATH_DEFINES
#include <math.h>

using namespace placeholders;

MultiSimp::MultiSimp(int n, dbl(*objectiveFunc)(int n, VectorXd &x), VectorXd& x, dbl Length, int timeout, dbl epn,int ncond, void(*Conds)(int n, VectorXd &x, int ncond, VectorXd &G), int nineq, void(*Ineqs)(int n, VectorXd &x, int nineq, VectorXd& H), int IterateNum, dbl ConvergenceCondition, VectorXd& Lambda, VectorXd& Mu){
	NCOND = ncond;
	calc_conds = Conds;
	NINEQ = nineq;
	calc_ineqs = Ineqs;
	ITERATETION_NUM = IterateNum;
	objective = objectiveFunc;
	nvar = n;
	optimizedVec = x;
	ConditionVal = ConvergenceCondition;
	ALPHA = 10.0; BETA = 0.25; C = FLT_MAX;
	length = Length;
	timeoutn = timeout;
	epsn = epn;
	Set_Nelder_Mead(nvar, std::bind(&MultiSimp::LagrangeFunc, this, _1, _2), optimizedVec, length, timeoutn, epsn);
	
}
MultiSimp::~MultiSimp() {
	cout << "close\n";
}
void vectorfprint_stderr(VectorXd &vec) {

	for (int I = 0; I < vec.size(); I++) {
		//cout << vec(I) << "\t";
		fprintf_s(stderr, "%.5e  ", vec(I));
		if ((I + 1) % 5 == 0) fprintf_s(stderr, "\n");//cout << "\n";
	}
	cout << "\n";
}
void MultiSimp::Initialize() {
	mu = VectorXd::Zero(NINEQ); lm = VectorXd::Zero(NCOND);
	s = 10.0*VectorXd::Ones(NINEQ); r = 10.0*VectorXd::Ones(NCOND);
	CondsVal = VectorXd::Zero(NCOND);
	gvalue = VectorXd::Zero(NCOND);
	IneqsVal = VectorXd::Zero(NINEQ);
	hvalue = VectorXd::Zero(NINEQ);
	
}
dbl MultiSimp::LagrangeFunc(int n, VectorXd &x) {
	dbl sum;
	sum = (*objective)(n, x)*PotentialWeight;
	(*calc_ineqs)(n, x, NINEQ ,IneqsVal);
	VectorXd VEC = s.cwiseProduct(IneqsVal) + mu;
	for (int i = 0; i < NINEQ; i++) {
		if (VEC(i) >= 0.0) {
			sum += (mu(i) + 0.5*s(i)*IneqsVal(i))*IneqsVal(i);
		}else {
			sum += (-0.5)*mu(i)*mu(i) / s(i);
		}
	}
	(*calc_conds)(n, x, NCOND,CondsVal);
	sum += lm.dot(CondsVal) + 0.5*r.cwiseProduct(CondsVal).dot(CondsVal);
	return sum;
}

void MultiSimp::Set_MultiSimpParams(int n, dbl(*objectiveFunc)(int n, VectorXd &x), int ncond, void(*Conds)(int n, VectorXd &x, int ncond,VectorXd &G), int nineq, void(*Ineqs)(int n, VectorXd &x, int nineq, VectorXd& H), VectorXd& x, dbl length, int IterateNum, dbl ConvergenceCondition, VectorXd& Lambda, VectorXd& Mu, int timeoutn, dbl epn) {
	NCOND = ncond;
	calc_conds = Conds;
	NINEQ = nineq;
	calc_ineqs = Ineqs;
	ITERATETION_NUM = IterateNum;
	objective = objectiveFunc;
	nvar = n;
	optimizedVec = x;
	ConditionVal = ConvergenceCondition;
	ALPHA = 10.0; BETA = 0.25; C = FLT_MAX;
}

status MultiSimp::Launch_MultiSimp(string FILENAME, void(*printFunc)(string filename)) {
	Initialize();
	int SIM_ITERATION;
	dbl lopt;
	dbl betac;
	status stat = failure;
	CGnuplot gp;
	set_printParams(FILENAME, printFunc);
	for (SIM_ITERATION = 0; SIM_ITERATION < ITERATETION_NUM; SIM_ITERATION++) {
		cout << "======================" << SIM_ITERATION << "-th iteration (multi-simp)=======================\n";
		//vector<dbl>().swap(ga); vector<dbl>().swap(ha);

		Set_Nelder_Mead(nvar, std::bind(&MultiSimp::LagrangeFunc, this, _1, _2), optimizedVec, length, timeoutn, epsn);
		status N_STATUS = launch_NelderMead(lopt,SIM_ITERATION);
		if (N_STATUS == failure) {
			stat = N_STATUS; break;
		}
		else {
			//cout << "==========================================================================\a\n";
			optimizedVec = return_vec();
			calc_conds(nvar, optimizedVec, NCOND, CondsVal);
			calc_ineqs(nvar, optimizedVec, NINEQ, IneqsVal);

			
			cout << " xopt\n";
			vectorfprint_stderr(optimizedVec);

			cout << "----------------\n";
			cout << " conds\n";
			vectorfprint_stderr(CondsVal);

			cout << "----------------\n";
			cout << " ineqs\n";
			vectorfprint_stderr(IneqsVal);
			VectorXd GA,HA;
			GA = IneqsVal.cwiseMax(-mu.cwiseQuotient(s)).cwiseAbs();
			HA = CondsVal.cwiseAbs();
			
			dbl gmax, hmax;
			gmax = GA.maxCoeff();
			hmax = HA.maxCoeff();

			if (gmax <= C && hmax <= C) {
				C = fmax(gmax, hmax);
				if (C < ConditionVal) {
					//C = fmax(gmax, hmax);
					stat = success;
					break;
				}
				VectorXd TMP = mu + s.cwiseProduct(IneqsVal);
				mu = TMP.cwiseMax(VectorXd::Zero(NINEQ));
				//for (int I = 0; I < NINEQ; I++) { mu(I) = fmax(0.0, TMP(I)); }
				lm = r.cwiseProduct(CondsVal);
			}
			cout << "===========MultipulterforConds===========\n";
			vectorfprint_stderr(lm);
			cout << "===========MultipulterforIneqs===========\n";
			vectorfprint_stderr(mu);
			cout << "==========================================\n";
			cout << "objective function->" << objective(nvar, optimizedVec) << "\n";
			cout << "c = " << C << "\n";

			betac = BETA * C;
			for (int j = 0; j < NINEQ; j++) { if (GA(j) > betac)  s(j) *= ALPHA; }
			for (int j = 0; j < NCOND; j++) { if (HA(j) > betac)  r(j) *= ALPHA; }
		}
	}
	
	return stat;
}