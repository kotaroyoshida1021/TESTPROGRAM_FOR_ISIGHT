#pragma once
#include "pch.h"
#include <Eigen/Core>
#include <vector>
#define MY_DEBUG_MODE
//static const int NDIV = 501;//�������C�z��v�Z�Ɏg�p
#define NDIV 251
static const int vbase = 251;//���֐��̌�
constexpr int dim = vbase;
#define NCOORD NDIV+3
#define U_NCOORD vbase
#define Xi_NCOORD vbase
#define Eta_NCOORD vbase
#define Zeta_NCOORD vbase
constexpr int NCOORD_PER_SURFACE = 3 * vbase + 5;
#define n_dim NCOORD_PER_SURFACE//�W���x�N�g���̑��a�̎���
#define NCOND 6//��������̌��Ȃ����1�ŁCCOND=0�������Ă���
#define NINEQ 42//�s��������̌��Ȃ����1�ŁCINEQ=-1�������Ă���
constexpr int bar_dim = 30;//�o���A�֐��ɂ�����]���_�̌�
static VectorXd AlphaParams;
static VectorXd OmgEtaParams;
static VectorXd DistParams;
static VectorXd coef;
static dbl OMG_ETA0, OMG_ETA_COEF;
static vector<dbl> BETA, OMG_ETA, DIST;//�q���x���g��ԗp
static VectorXd S;
static Vector2d Xi0Vec;
static vector<dbl> detPartDiff(9);
static vector<MatrixXd, Eigen::aligned_allocator<MatrixXd>> KinvPartdiff(9);

static dbl length_LL = 1.076991;//���C���[����
static dbl Ds = length_LL / (dbl)(NDIV - 1);//���ݕ�
static GaussIntegral<Vector3d> VecIntergalFunc;//�x�N�g���֐��ϕ���֐�
static GaussIntegral<dbl> ScalarIntegralFunc;//�X�J���[�֐��ϕ���֐�
static VectorFunction PosFuncL_str,PosFuncU_str;//�Ȑ��̈ʒu�x�N�g���̐��`�⊮�֐�
static ScalarFunction beta;
/*
> Ritz�@�ł���킳���݌v�ϐ��Q
*/
static VectorXd CondsVal, IneqsVal;//��������/�s��������̒l���i�[����x�N�g��
static VectorXd Lambda = VectorXd::Zero(NCOND);//�搔�@�ɂ����郉�O�����W���搔
static VectorXd Mu = VectorXd::Zero(NINEQ);//�搔�@�ɂ����郉�O�����W���搔