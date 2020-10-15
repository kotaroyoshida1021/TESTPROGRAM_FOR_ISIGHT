#pragma once
#include "pch.h"
#include <Eigen/Core>
#include <vector>
static const int NDIV = 501;//分割数，配列計算に使用
static const int vbase = 12;//基定関数の個数
constexpr int dim = vbase;
#define U_NCOORD vbase
#define Xi_NCOORD vbase
#define Eta_NCOORD vbase
#define Zeta_NCOORD vbase
constexpr int NCOORD_PER_SURFACE = 3 * vbase + 5;
#define n_dim NCOORD_PER_SURFACE//係数ベクトルの総和の次元
#define NCOND 3//等式制約の個数なければ1で，COND=0を加えておく
#define NINEQ 42//不等式制約の個数なければ1で，INEQ=-1を加えておく
constexpr int bar_dim = 30;//バリア関数における評価点の個数
static VectorXd AlphaParams;
static VectorXd OmgEtaParams;
static VectorXd DistParams;
static VectorXd Coef;
static VectorXd AlphaCoef, OmgEtaCoef, DistCoef;
static VectorXd S;
static Vector2d Xi0Vec;

static dbl length_LL = 1.076991;//ワイヤー長さ
static dbl Ds = length_LL / (dbl)(NDIV - 1);//刻み幅
static GaussIntegral<Vector3d> VecIntergalFunc;//ベクトル関数積分器関数
static GaussIntegral<dbl> ScalarIntegralFunc;//スカラー関数積分器関数
static VectorFunction PosFuncL_str,PosFuncU_str;//曲線の位置ベクトルの線形補完関数
/*
> Ritz法であらわされる設計変数群
*/
static VectorXd CondsVal, IneqsVal;//等式制約/不等式制約の値を格納するベクトル
static VectorXd Lambda = VectorXd::Zero(NCOND);//乗数法におけるラグランジュ乗数
static VectorXd Mu = VectorXd::Zero(NINEQ);//乗数法におけるラグランジュ乗数