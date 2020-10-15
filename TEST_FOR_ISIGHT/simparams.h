#pragma once
#include "pch.h"
#include <Eigen/Core>
#include <vector>
static const int NDIV = 501;//�������C�z��v�Z�Ɏg�p
static const int vbase = 12;//���֐��̌�
constexpr int dim = vbase;
#define U_NCOORD vbase
#define Xi_NCOORD vbase
#define Eta_NCOORD vbase
#define Zeta_NCOORD vbase
constexpr int NCOORD_PER_SURFACE = 3 * vbase + 5;
#define n_dim NCOORD_PER_SURFACE//�W���x�N�g���̑��a�̎���
#define NCOND 3//��������̌��Ȃ����1�ŁCCOND=0�������Ă���
#define NINEQ 42//�s��������̌��Ȃ����1�ŁCINEQ=-1�������Ă���
constexpr int bar_dim = 30;//�o���A�֐��ɂ�����]���_�̌�
static VectorXd AlphaParams;
static VectorXd OmgEtaParams;
static VectorXd DistParams;
static VectorXd Coef;
static VectorXd AlphaCoef, OmgEtaCoef, DistCoef;
static VectorXd S;
static Vector2d Xi0Vec;

static dbl length_LL = 1.076991;//���C���[����
static dbl Ds = length_LL / (dbl)(NDIV - 1);//���ݕ�
static GaussIntegral<Vector3d> VecIntergalFunc;//�x�N�g���֐��ϕ���֐�
static GaussIntegral<dbl> ScalarIntegralFunc;//�X�J���[�֐��ϕ���֐�
static VectorFunction PosFuncL_str,PosFuncU_str;//�Ȑ��̈ʒu�x�N�g���̐��`�⊮�֐�
/*
> Ritz�@�ł���킳���݌v�ϐ��Q
*/
static VectorXd CondsVal, IneqsVal;//��������/�s��������̒l���i�[����x�N�g��
static VectorXd Lambda = VectorXd::Zero(NCOND);//�搔�@�ɂ����郉�O�����W���搔
static VectorXd Mu = VectorXd::Zero(NINEQ);//�搔�@�ɂ����郉�O�����W���搔