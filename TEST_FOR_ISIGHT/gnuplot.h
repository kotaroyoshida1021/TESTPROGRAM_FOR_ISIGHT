#pragma once

/*! @file
 @brief		gnuplot�N���X
 @author		hecomi
 @date		July 5, 2010.
*/

#include <vector>
#include <fstream>
#include <string>

#if 0
class printFunctions {
private:
	std::string outputFile;
	void (*printFunc)(std::string outputFile);
public:
	//printFunctions();
	//printFunctions(std::string datafile, void(*print)(std::string)) {
		//outputFile = datafile;
		//printFunc = print;
	//};
	void setting(std::string datafile, void(*print)(std::string)) {
		outputFile = datafile;
		printFunc = print;
	};
	~printFunctions();
	printFunctions(const printFunctions& priFunc) :outputFile(priFunc.outputFile), printFunc(priFunc.printFunc) {
		//	fprintf_s(stderr, "Objective copied\n");
	}
};
#endif
class CGnuplot {
private:
	/*! @brief �p�C�v���q����t�@�C���|�C���^ */
	FILE* Fp;

	/*! @brief �o�b�t�@�t���b�V�� */
	void Flush();

public:
	/*! @brief �R���X�g���N�^ */
	CGnuplot();
	CGnuplot(const char* file_name);

	/*! @brief �f�X�g���N�^ */
	~CGnuplot();
	void EXIT_FUNC() {
		_pclose(Fp);
	}

	/*! @brief �ꎞ�t�@�C���� */
	static const std::string TempFileName;

	/*! @brief ����ɋ@�\���Ă��邩�ǂ��� */
	bool Check();

	/*! @brief printf���C�N�Ɏw��̃R�}���h�����s */
	void Command(const char* format, ...);

	/*! @brief �֐���`�� */
	void DrawFunc(const char* format);

	/* �v���b�g�^�C�v */
	static const int PLOT_TYPE_NOOUTPUT = 0;	//< �v���b�g���Ɉꎞ�t�@�C�����쐬���Ȃ�
	static const int PLOT_TYPE_OUTPUT = 1;		//< �v���b�g���Ɉꎞ�t�@�C�����쐬

	/*! @brief 1�����v�f��`��
	@param[in] cont �v���b�g�Ώ̃R���e�i
	@param[in] plot_type ����gnuplot�Ƀf�[�^��ł�����(0: default)���C�ꎞ�t�@�C�����쐬���邩(1: ������)�̑I��
	*/
	/*
	template <class T, template <class A, class Allocator = std::allocator<A> > class Container>
	void Plot(Container<T> cont, const int plot_type = PLOT_TYPE_OUTPUT, const char* file_name = TempFileName.c_str())
	{
		Container<T>::iterator it = cont.begin();

		switch (plot_type) {
		case PLOT_TYPE_NOOUTPUT:
			Command("plot '-' w lp");
			while (it != cont.end()) {
				Command("%f", *it);
				it++;
			}
			Command("e");
			break;

		case PLOT_TYPE_OUTPUT:
			std::ofstream fout(file_name);
			while (it != cont.end()) {
				fout << *it << std::endl;
				it++;
			}
			Command("plot '%s' w lines", file_name);
			break;
		}
	}
	*/
	/*! @brief 1�����v�f��`��i�z��j */
	/*
	template <class T, int N>
	void Plot(T(&cont)[N], const int plot_type = PLOT_TYPE_OUTPUT, const char* file_name = TempFileName.c_str())
	{
		int x = 0;

		switch (plot_type) {
		case PLOT_TYPE_NOOUTPUT:
			Command("plot '-' w lp");
			while (x < N) {
				Command("%f", cont[x]);
				x++;
			}
			Command("e");
			break;

		case PLOT_TYPE_OUTPUT:
			std::ofstream fout(file_name);
			while (x < N) {
				fout << cont[x] << std::endl;
				x++;
			}
			Command("plot '%s' w lines", file_name);
			break;
		}
	}
	*/
	/*! @brief 2�����v�f��`�� */
	/*
	template <class T, template <class A, class Allocator = std::allocator<A> > class Container>
	void Plot(Container<T> contX, Container<T> contY, const int plot_type = PLOT_TYPE_OUTPUT, const char* file_name = TempFileName.c_str())
	{
		Container<T>::iterator itX = contX.begin();
		Container<T>::iterator itY = contY.begin();
		switch (plot_type) {
		case PLOT_TYPE_NOOUTPUT:
			Command("plot '-' w lp");
			while (itX != contX.end() && itY != contY.end()) {
				Command("%f %f", *itX, *itY);
				itX++; itY++;
			}
			Command("e");
			break;

		case PLOT_TYPE_OUTPUT:
			std::ofstream fout(file_name);
			while (itX != contX.end() && itY != contY.end()) {
				fout << *itX << " " << *itY << std::endl;
				itX++; itY++;
			}
			Command("plot '%s' w lines", file_name);
			break;
		}
	}
	*/
	/*! @brief 2�����v�f��`��i�z��j */
	/*
	template <class T, int N, int M>
	void Plot(T(&contX)[N], T(&contY)[M], const int plot_type = PLOT_TYPE_NOOUTPUT, const char* file_name = TempFileName.c_str())
	{
		int x = 0, y = 0;
		switch (plot_type) {
		case PLOT_TYPE_NOOUTPUT:
			Command("plot '-' w lp");
			while (x < N && y < M) {
				Command("%f %f", contX[x], contY[y]);
				x++; y++;
			}
			Command("e");
			break;

		case PLOT_TYPE_OUTPUT:
			std::ofstream fout(file_name);
			while (x < N && y < M) {
				fout << contX[x] << " " << contY[y] << std::endl;
				x++; y++;
			}
			Command("plot '%s' w lines", file_name);
			break;
		}
	}
	*/
	/*! @brief X���x�����Z�b�g */
	void SetXLabel(const char* format);

	/*! @brief Y���x�����Z�b�g */
	void SetYLabel(const char* format);

	/*! @brief X�v���b�g�͈͂�ݒ� */
	void SetXRange(const double x_min, const double x_max);

	/*! @brief Y�v���b�g�͈͂�ݒ� */
	void SetYRange(const double y_min, const double y_max);

	/*! @brief ���v���b�g */
	void Replot();
};