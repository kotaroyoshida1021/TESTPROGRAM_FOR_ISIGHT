#pragma once
#include <iostream>
#include "pch.h"
#include <Eigen/Core>
#include <vector>
#include "Obj_Coordinates.h"
#include <map>
/*
 > class of calculating developable surface infomation from given two curves
 > ****member****
 > obj_L,obj_U : given by user
 > ALPHA,..DEVELOPABLE_CONDITION : arrays for storing calculated datas
 > alpha_str,...eta_str : a function to interpolate array
 > ****function****
 > SurfaceInfo : a contractor
 > UpdateSurfaceInfo : main function, calculating alpha etc.
 > UpdateInput : Update imformation of two curves
 > terminates : terminate ALPHA,..etc.
 */
class SurfaceInfo {
public:
	Coordinates obj_L, obj_U;
	vector<dbl> ALPHA, LMD, OMG_XI, DEVELOPABLE_CONDTION,DIST;
	vector<Vector3d, aligned_allocator<Vector3d>> GENERATRIX, XI, ETA;
	SurfaceInfo(Coordinates &objectCoordinates_L, Coordinates &objectCoodinates_U);
	void UpdateSurfaceInfo();
	ScalarFunction alpha_str, lambda_str, omg_Xi_str, DevelopableCondition_str,dist_str;
	VectorFunction generatrix_str, xi_str, eta_str;
	void UpdateInput(Coordinates& objectCoordinates_L, Coordinates& objectCoodinates_U);
	~SurfaceInfo();
	void terminates();
};