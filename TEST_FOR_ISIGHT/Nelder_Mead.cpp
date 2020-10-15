#include "pch.h"
#include <iostream>
#include <Eigen/Core>
#include <vector>
#include "Nelder_Mead.h"
#include "gnuplot.h"
#include <map>

/*
Nelder_Mead::Nelder_Mead(int n, function<dbl(int,VectorXd&)> func, VectorXd initvec, dbl length, int timeout,dbl eps) {
	nvar = n;
	obj = func;
	x_init = initvec;
	L = length;
	EPS = eps;
	TIMEOUT = timeout;

}
*/
void Nelder_Mead::Set_Nelder_Mead(int n, function<dbl(int, VectorXd&)> func, VectorXd &initvec, dbl length, int timeout, dbl eps) {
	nvar = n;
	obj = func;
	x_init = initvec;
	L = length;
	EPS = eps;
	TIMEOUT = timeout;
}

void Nelder_Mead::initializing_num() {
	n_contract = 0; n_expand = 0; n_inside = 0; n_reflect = 0; n_shrink = 0;
}

void Nelder_Mead::SetAdaptiveParams(int n){
	alpha = 1.0; beta = 1.0 + 2.0 / (dbl)(n); gamma = 0.75 - 1.0 / (2.0*(dbl)(n)); delta = 1.0 - 1.0 / (dbl)(n);
}

void Nelder_Mead::initialize() {
	VectorXd e = VectorXd::Zero(nvar);
	VectorXd ret;
	dbl dl_1, dl_2;
	dl_1 = (sqrt(nvar + 1.0) + nvar - 1.0) / (sqrt(2.0)*nvar);
	dl_2 = (sqrt(nvar + 1.0) - 1.0) / (sqrt(2.0)*nvar);
	for (int j = 0; j <= nvar; j++) {
		if(j!=0) {
			e = dl_2 * VectorXd::Ones(nvar);
			e(j - 1) = dl_1;
		}
		ret = L * e + x_init;
		dbl R = obj(nvar, ret);
		//simplex.push_back(ObjectiveFunction(ret, R));
		simplex.push_back(make_pair(R,ret));
	}
}
bool OnlyFirstComparing(const pair<dbl, VectorXd>& P1, const pair<dbl, VectorXd>& P2) {
	return P1.first < P2.first;
}
void Nelder_Mead::SearchSimplex() {
	//std::sort(simplex.begin(), simplex.end(), ObjectiveFunction::comparing);
	sort(simplex.begin(), simplex.end(),OnlyFirstComparing);
}

void Nelder_Mead::calc_Centroid() {
	//x_centroid = simplex[0].a;
	x_centroid = simplex[0].second;
	for (int i = 1; i < nvar; i++) {
		x_centroid += simplex[i].second;
	}
	x_centroid /= nvar;
}

dbl Nelder_Mead::calc_variance() {
	dbl fmean,fvar;
	fmean = 0.0; fvar = 0.0;
	for (int i = 0; i <= nvar; i++) {
		fmean += simplex[i].first;
	}
	fmean /= (dbl)(nvar + 1);
	for (int i = 0; i <= nvar; i++) {
		fvar += (simplex[i].first - fmean)*(simplex[i].first - fmean);
	}
	fvar /= (dbl)(nvar + 1);
	return fvar;
}

void Nelder_Mead::shrink() {
	VectorXd x0;
	x0 = simplex[0].second;
	simplex[0].first = obj(nvar, x0);
	for (int i = 1; i <= nvar; i++) {
		simplex[i].second = x0 + delta * (simplex[i].second - x0);
		simplex[i].first = obj(nvar, simplex[i].second);
	}
	n_shrink++;
}

void Nelder_Mead::fprintCoeffcients() {
	cout << simplex[0].second << "\n";
}
status Nelder_Mead::launch_NelderMead(dbl& fopt,int Iterate_num) {
	status stat;
	initializing_num();
	SetAdaptiveParams(nvar);
	initialize();
	//ObjectiveFunction _reflect(x_init,0.0), _contract(x_init,0.0), _expand(x_init,0.0), _inside(x_init,0.0);
	pair<dbl, VectorXd> _reflect, _contract, _expand, _inside;
	CGnuplot gp;
	stat = failure;
	for (int j = 0; j <= nvar; j++) {
		simplex[j].first = obj(nvar, simplex[j].second);
	}
	
	cout << "   count| reflection| expansion|contraction|   inside |  shrink | fvalue" << "\n";
	for (int i = 0; i < TIMEOUT; i++) {
		SearchSimplex();
		calc_Centroid();
		if (calc_variance() < EPS) {
			stat = success;
			break;
		}
		if (i%SKIPTIME == 0) {
			fprintf_s(stderr, "\r%8d|  %8d |  %7d |  %7d  | %8d | %7d | %6.3e \r", i, n_reflect, n_expand, n_contract, n_inside, n_shrink, simplex[0].first);
			if (simplex[0].first > UPPERfvalue) {
				cout << "\n";
				cout << "===============================================" << "\n";
				cout << " exceed an upper limit fvalue " << UPPERfvalue << ".EXIT." << "\n";
				cout << "===============================================" << "\n"; break;
			}
		}
		if (i%MONITORTIME == 0) {
			ostringstream ss,si;
			ss << i;
			si << Iterate_num;
			printFunc(filename);
			string command = "splot \"" + filename + "\" u 1:2:3 w lp pt 2 t \"k = " + ss.str() + " in Iteration " + si.str() + "-th\" \n"; //+ ",\"input.txt\" with points" + ",\"inputL.txt\" with points\n";
#if SetGPMonitor
			gp.Command(command.c_str());
#endif
		}
		_reflect.second = (1.0 + alpha)*x_centroid - alpha * simplex[nvar].second;
		_reflect.first = obj(nvar, _reflect.second);
		n_reflect++;
		if (_reflect.first <= simplex[nvar - 1].first) {
			if (_reflect.first >= simplex[0].first) {
				simplex[nvar] = _reflect;
			}
			else {
				_expand.second = (beta)*_reflect.second + (1.0 - beta)*x_centroid;
				_expand.first = obj(nvar, _expand.second);
				n_expand++;
				if (_expand.first < simplex[0].first) {
					simplex[nvar] = _expand;
				}
				else {
					simplex[nvar] = _reflect;
				}
			}
		}
		else {
			if (_reflect.first < simplex[nvar].first) {
				_contract.second = (1.0 - gamma)*x_centroid + gamma * _reflect.second;
				_contract.first = obj(nvar, _contract.second);
				n_contract++;
				if (_contract.first <= _reflect.first) {
					simplex[nvar] = _contract;
				}
				else shrink();

			}
			else {
				_inside.second = -gamma * _reflect.second + (1.0 + gamma)*x_centroid;
				_inside.first = obj(nvar, _inside.second);
				n_inside++;
				if (_inside.first <= simplex[nvar].first) {
					simplex[nvar] = _inside;
				}
				else shrink();
			}
		}
	}
	x_init = simplex[0].second; 
	fopt = obj(nvar, simplex[0].second);
	cout << "\n";
	cout << "==========================================================================\a\n";
	gp.~CGnuplot();
	return stat;
}
