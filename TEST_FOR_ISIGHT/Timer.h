#pragma once
#include <windows.h>
#pragma comment(lib, "winmm.lib")
class Timer
{
public:
	Timer() { restart(); }
public:
	void  restart()
	{
		m_start = timeGetTime();        // �v���J�n���Ԃ�ۑ�
	}
	double  elapsed()    // ���X�^�[�g����̕b����Ԃ�
	{
		DWORD end = timeGetTime();
		return (double)(end - m_start) / 1000;
	}
	int hour() {
		double tmp = elapsed();
		return (int)(tmp / 3600);
	}
	int minute() {
		double tmp = elapsed();
		return (int)((tmp - 3600 * hour()) / 60);
	}
	double sec() {
		double tmp = elapsed();
		int min = minute(), hours = hour();
		dbl d_min = (double)(min * 60);
		dbl d_hour = (double)(hours * 3600);
		return tmp - d_min - d_hour;
	}
	void print_times() {
		std::cout << "#" << hour() << "[h.]" << minute() << "[min.]" << sec() << "[sec.]" << "\n";
	}
private:
	DWORD    m_start;    //  �v���J�n����
};