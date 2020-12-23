#pragma once
#include "pch.h"
#include <Eigen/Core>
#include <vector>
//#define MY_DEBUG_MODE
#define	alloc(type,size) (type *)malloc((size)*sizeof(type))
//static const int NDIV = 501;//分割数，配列計算に使用
#define NDIV 251//THE QUANTITY OF DIVISION USING DISCRETIZATION
static constexpr int NCOND = 6, NINEQ = NDIV;//THE QUANTITY OF CONDITIONS AND INEQUALITIES
static VectorXd AlphaParams, OmgEtaParams, DistParams; //DETERMINE IN FUNCTION:initializing

static MatrixXd GramA, GramE, GramD, KerA, KerD, KerE;//DETERMINE IN FUNCTION:determineDimension()
static int RankA, RankE, RankD, NCOORD_ALPHA, NCOORD_ETA, NCOORD_DIST,NCOORD;//RANK OF GRAM_MATRIX AND THE DIMENSION OF COEF IN EACH FUNCTION->DETERMINE IN FUNCTION:determineDimension()
const string file_name = "test20201113vbase8V3";
static dbl length_LL = 1.076991;//ワイヤー長さ
static dbl Ds = length_LL / (dbl)(NDIV - 1);//刻み幅
static GaussIntegral<Vector3d> VecIntergalFunc;//ベクトル関数積分器関数
static GaussIntegral<dbl> ScalarIntegralFunc;//スカラー関数積分器関数
static VectorFunction PosFuncL_str, PosFuncU_str;//曲線の位置ベクトルの線形補完関数
static ScalarFunction beta, alpha, omegaEta, dist;
static const int vbase = 8;//基定関数の個数
static RitzMethod Bt(vbase, length_LL);
static RitzMethod forOMGET(vbase, length_LL), forDIST(vbase, length_LL);

constexpr int dim = vbase;
static int NCOORD_PER_SURFACE = 3 * vbase + 5;
#define n_dim NCOORD_PER_SURFACE//係数ベクトルの総和の次元
//等式制約の個数なければ1で，COND=0を加えておく
//不等式制約の個数なければ1で，INEQ=-1を加えておく

static VectorXd coef;
static VectorXd PARAM_A, PARAM_E, PARAM_D;
static VectorXd BETA_H, OMG_ETA_H, DIST_H;//ヒルベルト空間用
static vector<dbl> BETA(NDIV), OMG_ETA(NDIV), DIST(NDIV), ALPHA(NDIV - 2);//線形補間用
static Vector2d Xi0Vec;
static CGnuplot gp;

static vector<dbl> EigVal;
/* > Ritz法であらわされる設計変数群 */
static VectorXd CondsVal, IneqsVal;//等式制約/不等式制約の値を格納するベクトル
static VectorXd Lambda = VectorXd::Zero(NCOND);//乗数法におけるラグランジュ乗数
static VectorXd Mu = VectorXd::Zero(NINEQ);//乗数法におけるラグランジュ乗数
