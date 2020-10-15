#pragma once
#include "pch.h"
#include <iostream>
#include "Obj_Coordinates.h"

class BisectionMethod {
	dbl eps = 1.0e-7;
	static const int limit_Num = 100000;
public:
	dbl init, end;
	dbl Solution(dbl(*func)(dbl s), dbl INIT, dbl END);
};