#pragma once
#include "pch.h"
#include <iostream>
#include <Eigen/Core>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace Eigen;
class RitzMethod {
private:
	vector<VectorXd, Eigen::aligned_allocator<VectorXd>> BaseFunctions;
	vector<VectorXd, Eigen::aligned_allocator<VectorXd>> BaseFunctionsdot;
	dbl length;
public:
	RitzMethod(int nvar,dbl Length);//nvarÇÕåWêîÇÃêî,LengthÇ‡ìnÇ∑
	VectorXd a;
	dbl Function(dbl);
	dbl operator()(dbl s);
	dbl Partial(int i, dbl s);
	void terminates();
	dbl Derivative(dbl s);
};

static const int BMAX = 10001;

static dbl step(dbl q, dbl s) { return 1.0; }
static dbl ramp(dbl q, dbl s) { return q * s / M_PI; }

static dbl sin1(dbl q, dbl s) { return sin(1 * q*s); }
static dbl sin2(dbl q, dbl s) { return sin(2 * q*s); }
static dbl sin3(dbl q, dbl s) { return sin(3 * q*s); }
static dbl sin4(dbl q, dbl s) { return sin(4 * q*s); }
static dbl sin5(dbl q, dbl s) { return sin(5 * q*s); }
static dbl sin6(dbl q, dbl s) { return sin(6 * q*s); }
static dbl sin7(dbl q, dbl s) { return sin(7 * q*s); }
static dbl sin8(dbl q, dbl s) { return sin(8 * q*s); }
static dbl sin9(dbl q, dbl s) { return sin(9 * q*s); }
static dbl sin10(dbl q, dbl s) { return sin(10 * q*s); }
static dbl sin11(dbl q, dbl s) { return sin(11 * q*s); }
static dbl sin12(dbl q, dbl s) { return sin(12 * q*s); }
static dbl sin13(dbl q, dbl s) { return sin(13 * q*s); }
static dbl sin14(dbl q, dbl s) { return sin(14 * q*s); }
static dbl sin15(dbl q, dbl s) { return sin(15 * q*s); }

static dbl cos1(dbl q, dbl s) { return cos(1 * q*s); }
static dbl cos2(dbl q, dbl s) { return cos(2 * q*s); }
static dbl cos3(dbl q, dbl s) { return cos(3 * q*s); }
static dbl cos4(dbl q, dbl s) { return cos(4 * q*s); }
static dbl cos5(dbl q, dbl s) { return cos(5 * q*s); }
static dbl cos6(dbl q, dbl s) { return cos(6 * q*s); }
static dbl cos7(dbl q, dbl s) { return cos(7 * q*s); }
static dbl cos8(dbl q, dbl s) { return cos(8 * q*s); }
static dbl cos9(dbl q, dbl s) { return cos(9 * q*s); }
static dbl cos10(dbl q, dbl s) { return cos(10 * q*s); }
static dbl cos11(dbl q, dbl s) { return cos(11 * q*s); }
static dbl cos12(dbl q, dbl s) { return cos(12 * q*s); }
static dbl cos13(dbl q, dbl s) { return cos(13 * q*s); }
static dbl cos14(dbl q, dbl s) { return cos(14 * q*s); }
static dbl cos15(dbl q, dbl s) { return cos(15 * q*s); }

static dbl stepdot(dbl q, dbl s) { return 0; }
static dbl rampdot(dbl q, dbl s) { return q / M_PI; }

static dbl sin1dot(dbl q, dbl s) { return 1 * q*cos(1 * q*s); }
static dbl sin2dot(dbl q, dbl s) { return 2 * q*cos(2 * q*s); }
static dbl sin3dot(dbl q, dbl s) { return 3 * q*cos(3 * q*s); }
static dbl sin4dot(dbl q, dbl s) { return 4 * q*cos(4 * q*s); }
static dbl sin5dot(dbl q, dbl s) { return 5 * q*cos(5 * q*s); }
static dbl sin6dot(dbl q, dbl s) { return 6 * q*cos(6 * q*s); }
static dbl sin7dot(dbl q, dbl s) { return 7 * q*cos(7 * q*s); }
static dbl sin8dot(dbl q, dbl s) { return 8 * q*cos(8 * q*s); }
static dbl sin9dot(dbl q, dbl s) { return 9 * q*cos(9 * q*s); }
static dbl sin10dot(dbl q, dbl s) { return 10 * q*cos(10 * q*s); }
static dbl sin11dot(dbl q, dbl s) { return 11 * q*cos(11 * q*s); }
static dbl sin12dot(dbl q, dbl s) { return 12 * q*cos(12 * q*s); }
static dbl sin13dot(dbl q, dbl s) { return 13 * q*cos(13 * q*s); }
static dbl sin14dot(dbl q, dbl s) { return 14 * q*cos(14 * q*s); }
static dbl sin15dot(dbl q, dbl s) { return 15 * q*cos(15 * q*s); }

static dbl cos1dot(dbl q, dbl s) { return -1 * q*sin(1 * q*s); }
static dbl cos2dot(dbl q, dbl s) { return -2 * q*sin(2 * q*s); }
static dbl cos3dot(dbl q, dbl s) { return -3 * q*sin(3 * q*s); }
static dbl cos4dot(dbl q, dbl s) { return -4 * q*sin(4 * q*s); }
static dbl cos5dot(dbl q, dbl s) { return -5 * q*sin(5 * q*s); }
static dbl cos6dot(dbl q, dbl s) { return -6 * q*sin(6 * q*s); }
static dbl cos7dot(dbl q, dbl s) { return -7 * q*sin(7 * q*s); }
static dbl cos8dot(dbl q, dbl s) { return -8 * q*sin(8 * q*s); }
static dbl cos9dot(dbl q, dbl s) { return -9 * q*sin(9 * q*s); }
static dbl cos10dot(dbl q, dbl s) { return -10 * q*sin(10 * q*s); }
static dbl cos11dot(dbl q, dbl s) { return -11 * q*sin(11 * q*s); }
static dbl cos12dot(dbl q, dbl s) { return -12 * q*sin(12 * q*s); }
static dbl cos13dot(dbl q, dbl s) { return -13 * q*sin(13 * q*s); }
static dbl cos14dot(dbl q, dbl s) { return -14 * q*sin(14 * q*s); }
static dbl cos15dot(dbl q, dbl s) { return -15 * q*sin(15 * q*s); }

static dbl stepddot(dbl q, dbl s) { return 0; }
static dbl rampddot(dbl q, dbl s) { return 0; }

static dbl sin1ddot(dbl q, dbl s) { return -1 * q*q*sin(1 * q*s); }
static dbl sin2ddot(dbl q, dbl s) { return -4 * q*q*sin(2 * q*s); }
static dbl sin3ddot(dbl q, dbl s) { return -9 * q*q*sin(3 * q*s); }
static dbl sin4ddot(dbl q, dbl s) { return -16 * q*q*sin(4 * q*s); }
static dbl sin5ddot(dbl q, dbl s) { return -25 * q*q*sin(5 * q*s); }
static dbl sin6ddot(dbl q, dbl s) { return -36 * q*q*sin(6 * q*s); }
static dbl sin7ddot(dbl q, dbl s) { return -49 * q*q*sin(7 * q*s); }
static dbl sin8ddot(dbl q, dbl s) { return -64 * q*q*sin(8 * q*s); }
static dbl sin9ddot(dbl q, dbl s) { return -81 * q*q*sin(9 * q*s); }
static dbl sin10ddot(dbl q, dbl s) { return -100 * q*q*sin(10 * q*s); }
static dbl sin11ddot(dbl q, dbl s) { return -121 * q*q*sin(11 * q*s); }
static dbl sin12ddot(dbl q, dbl s) { return -144 * q*q*sin(12 * q*s); }
static dbl sin13ddot(dbl q, dbl s) { return -169 * q*q*sin(13 * q*s); }
static dbl sin14ddot(dbl q, dbl s) { return -196 * q*q*sin(14 * q*s); }
static dbl sin15ddot(dbl q, dbl s) { return -225 * q*q*sin(15 * q*s); }

static dbl cos1ddot(dbl q, dbl s) { return -1 * q*q*cos(1 * q*s); }
static dbl cos2ddot(dbl q, dbl s) { return -4 * q*q*cos(2 * q*s); }
static dbl cos3ddot(dbl q, dbl s) { return -9 * q*q*cos(3 * q*s); }
static dbl cos4ddot(dbl q, dbl s) { return -16 * q*q*cos(4 * q*s); }
static dbl cos5ddot(dbl q, dbl s) { return -25 * q*q*cos(5 * q*s); }
static dbl cos6ddot(dbl q, dbl s) { return -36 * q*q*cos(6 * q*s); }
static dbl cos7ddot(dbl q, dbl s) { return -49 * q*q*cos(7 * q*s); }
static dbl cos8ddot(dbl q, dbl s) { return -64 * q*q*cos(8 * q*s); }
static dbl cos9ddot(dbl q, dbl s) { return -81 * q*q*cos(9 * q*s); }
static dbl cos10ddot(dbl q, dbl s) { return -100 * q*q*cos(10 * q*s); }
static dbl cos11ddot(dbl q, dbl s) { return -121 * q*q*cos(11 * q*s); }
static dbl cos12ddot(dbl q, dbl s) { return -144 * q*q*cos(12 * q*s); }
static dbl cos13ddot(dbl q, dbl s) { return -169 * q*q*cos(13 * q*s); }
static dbl cos14ddot(dbl q, dbl s) { return -196 * q*q*cos(14 * q*s); }
static dbl cos15ddot(dbl q, dbl s) { return -225 * q*q*cos(15 * q*s); }

#define TNUM 28
static dbl(*base[TNUM])(dbl q, dbl s) = { step ,ramp ,
									sin1 ,cos1 ,sin2 ,cos2 ,sin3 ,cos3 ,sin4 ,cos4
									,sin5,cos5,sin6,cos6, sin7,cos7,sin8,cos8,sin9,cos9
									,sin10,cos10,sin11,cos11,sin12,cos12,sin13,cos13/**/ };
static dbl(*basedot[TNUM])(dbl q, dbl s) = { stepdot ,rampdot ,
								sin1dot ,cos1dot ,sin2dot ,cos2dot,sin3dot,cos3dot,sin4dot ,cos4dot
								,sin5dot,cos5dot ,sin6dot ,cos6dot,sin7dot,cos7dot,sin8dot,cos8dot,sin9dot,cos9dot
								,sin10dot,cos10dot,sin11dot,cos11dot,sin12dot,cos12dot,sin13dot,cos13dot/**/ };

