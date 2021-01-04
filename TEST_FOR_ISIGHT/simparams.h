#pragma once
#include "pch.h"
#include <Eigen/Core>
#include <vector>
//#define MY_DEBUG_MODE
#define	alloc(type,size) (type *)malloc((size)*sizeof(type))
//static const int NDIV = 501;//�������C�z��v�Z�Ɏg�p
#define NDIV 251//THE QUANTITY OF DIVISION USING DISCRETIZATION
static constexpr int NCOND = 6, NINEQ = NDIV;//THE QUANTITY OF CONDITIONS AND INEQUALITIES
static VectorXd AlphaParams, OmgEtaParams, DistParams; //DETERMINE IN FUNCTION:initializing

static MatrixXd GramA, GramE, GramD, KerA, KerD, KerE;//DETERMINE IN FUNCTION:determineDimension()
static int RankA, RankE, RankD, NCOORD_ALPHA, NCOORD_ETA, NCOORD_DIST,NCOORD;//RANK OF GRAM_MATRIX AND THE DIMENSION OF COEF IN EACH FUNCTION->DETERMINE IN FUNCTION:determineDimension()
const string file_name = "test20201113vbase8V4";
static dbl length_LL = 1.076991;//���C���[����
static dbl Ds = length_LL / (dbl)(NDIV - 1);//���ݕ�
static GaussIntegral<Vector3d> VecIntergalFunc;//�x�N�g���֐��ϕ���֐�
static GaussIntegral<dbl> ScalarIntegralFunc;//�X�J���[�֐��ϕ���֐�
static VectorFunction PosFuncL_str, PosFuncU_str;//�Ȑ��̈ʒu�x�N�g���̐��`�⊮�֐�
static ScalarFunction beta, alpha, omegaEta, dist, dev_conds;
static const int vbase = 8;//���֐��̌�
#define NCOORD_PER_PHI vbase
#define NCOORD_PER_THETA vbase
#define NCOORD_PER_UFUNC vbase
static RitzMethod phi(vbase, length_LL), theta(vbase, length_LL), uSdot(vbase, length_LL);

constexpr int dim = vbase;
static int NCOORD_PER_SURFACE = 3 * vbase + 5;
#define n_dim NCOORD_PER_SURFACE//�W���x�N�g���̑��a�̎���
//��������̌��Ȃ����1�ŁCCOND=0�������Ă���
//�s��������̌��Ȃ����1�ŁCINEQ=-1�������Ă���

static VectorXd coef;
static VectorXd PARAM_A, PARAM_E, PARAM_D;
static VectorXd BETA_H, OMG_ETA_H, DIST_H;//�q���x���g��ԗp
static vector<dbl> BETA(NDIV), OMG_ETA(NDIV), DIST(NDIV), ALPHA(NDIV), DEV_CONDS(NDIV);//���`��ԗp
static CGnuplot gp;

static vector<dbl> EigVal;
/* > Ritz�@�ł���킳���݌v�ϐ��Q */
static VectorXd CondsVal, IneqsVal;//��������/�s��������̒l���i�[����x�N�g��
static VectorXd Lambda = VectorXd::Zero(NCOND);//�搔�@�ɂ����郉�O�����W���搔
static VectorXd Mu = VectorXd::Zero(NINEQ);//�搔�@�ɂ����郉�O�����W���搔
