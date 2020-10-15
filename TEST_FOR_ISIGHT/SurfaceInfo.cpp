
#include <iostream>
#include "pch.h"
#include <Eigen/Eigen>
#include <vector>
#include "Obj_Coordinates.h"
#include <map>
#include "LinearFunction.h"
#include "SurfaceInfo.h"


using namespace std;
using namespace Eigen;
using namespace std::placeholders;

SurfaceInfo::SurfaceInfo(Coordinates& objectCoordinates_L, Coordinates& objectCoodinates_U) :obj_L(objectCoordinates_L), obj_U(objectCoodinates_U){
	//cout << "Ds = " << obj_L.Ds << ",Ds = " << obj_U.Ds << "\n";
}
void SurfaceInfo::UpdateInput(Coordinates& objectCoordinates_L, Coordinates& objectCoodinates_U) {
	obj_L = objectCoordinates_L;
	obj_U = objectCoodinates_U;
}
void SurfaceInfo::UpdateSurfaceInfo() {
	//cout << __func__ << "Started\n";
	Vector3d ZERO;
	ZERO << 0.0, 0.0, 0.0;
	ALPHA.push_back(0.0);
	GENERATRIX.push_back(ZERO);
	Vector3d tmp = obj_U.ZETA[0].cross(obj_L.ZETA[0]);
	ETA.push_back(tmp.normalized()); XI.push_back(-obj_L.ZETA[0].cross(ETA[0]));
	LMD.push_back(-obj_L.omg_Xi(0.0) * obj_L.ETA[0].dot(XI[0]) + obj_L.omg_Eta(0.0) * obj_L.XI[0].dot(XI[0]));
	OMG_XI.push_back(obj_L.omg_Xi(0.0) * obj_L.ETA[0].dot(ETA[0]) - obj_L.omg_Zeta(0.0) * obj_L.XI[0].dot(ETA[0]));
	DIST.push_back(0.0);
	if (obj_L.POS.size() != obj_U.POS.size()) {
		cout << "error;size not match in func Update\n";
		cout << "L:" << obj_L.POS.size() << "U:" << obj_U.POS.size() << "\n";
		exit(1);
	}
	for (unsigned int i = 1; i < obj_L.POS.size()-1; i++) {
		Vector3d diff = obj_U.POS[i] - obj_L.POS[i];
		dbl Ds = obj_L.Ds;
		DIST.push_back(diff.norm());
		GENERATRIX.push_back(diff.normalized());
		ALPHA.push_back(-asin(diff.dot(obj_L.ZETA[i])));
		XI.push_back(GENERATRIX[i] / cos(ALPHA[i]) + obj_L.ZETA[i] * tan(ALPHA[i]));
		ETA.push_back(obj_L.ZETA[i].cross(XI[i]));
		LMD.push_back(-obj_L.omg_Xi(i * Ds) * obj_L.ETA[i].dot(XI[i]) + obj_L.omg_Eta(i * Ds) * obj_L.XI[i].dot(XI[i]));
		OMG_XI.push_back(obj_L.omg_Xi(i * Ds) * obj_L.ETA[i].dot(ETA[i]) - obj_L.omg_Eta(i * Ds) * obj_L.XI[i].dot(ETA[i]));
	}
	int max = obj_L.POS.size() - 1;
	Vector3d diff = obj_U.POS[max] - obj_L.POS[max];
	ALPHA.push_back(ALPHA[obj_L.POS.size() - 2]); LMD.push_back(LMD[obj_L.POS.size() - 2]); OMG_XI.push_back(OMG_XI[obj_L.POS.size() - 2]);
	DIST.push_back(diff.norm());
	alpha_str.set_Info(obj_L.Ds, ALPHA); generatrix_str.set_Info(obj_L.Ds, GENERATRIX); eta_str.set_Info(obj_L.Ds, ETA);
	xi_str.set_Info(obj_L.Ds, XI); lambda_str.set_Info(obj_L.Ds, LMD); omg_Xi_str.set_Info(obj_L.Ds, OMG_XI);
	dist_str.set_Info(obj_L.Ds, DIST);

	for (int i = 0; i < obj_L.POS.size(); i++) {
		Vector3d diff,dn;
		diff = obj_U.POS[i] - obj_L.POS[i];
		dn = diff.normalized();
		tmp = obj_L.ZETA[i].cross(obj_U.ZETA[i]);
		DEVELOPABLE_CONDTION.push_back(fabs(dn.dot(tmp)));
	}
	DevelopableCondition_str.set_Info(obj_L.Ds, DEVELOPABLE_CONDTION);
	//cout << __func__ << " is done\n";
}

void SurfaceInfo::terminates() {
	vector<dbl>().swap(ALPHA); vector<dbl>().swap(LMD); vector<dbl>().swap(OMG_XI); vector<dbl>().swap(DEVELOPABLE_CONDTION);
	vector<Vector3d, aligned_allocator<Vector3d>>().swap(XI);
	vector<Vector3d, aligned_allocator<Vector3d>>().swap(ETA);
	vector<Vector3d, aligned_allocator<Vector3d>>().swap(GENERATRIX);
	vector<dbl>().swap(DIST);
}
SurfaceInfo::~SurfaceInfo() { terminates(); };
