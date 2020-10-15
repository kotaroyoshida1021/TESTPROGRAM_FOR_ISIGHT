#include "pch.h"
#include <stdio.h>
#include <iostream>
#include <stdarg.h>
#include "gnuplot.h"

const std::string CGnuplot::TempFileName = "temp.dat";
//static const char* GNUPLOT_PATH = "C:/Program\\ Files/gnuplot/bin/gnuplot.exe -persist";
//const char* GNUPLOT_PATH = "\"C:/Program Files (x86)/gnuplot/bin/gnuplot.exe\" -persist";
//const char* GNUPLOT_PATH = "\"C:/Program Files (x86)/gnuplot/bin/gnuplot.exe\"";
const char* GNUPLOT_PATH = "\"C:/Program Files/gnuplot/bin/gnuplot.exe\"";
//static const char* GNUPLOT_PATH = "\"C:/Program Files (x86)/gnuplot/bin/gnuplot.exe\" -persist";

//printFunctions::~printFunctions() {};

CGnuplot::CGnuplot(void)
{
	Fp = _popen(GNUPLOT_PATH, "w");
	if (Fp == NULL) {
		printf("pipe error\n");
	}
}

CGnuplot::CGnuplot(const char* file_name)
{
	fopen_s(&Fp,file_name, "w");
	if (Fp == NULL) {
		printf("pipe error\n");
	}
}

CGnuplot::~CGnuplot()
{
	_pclose(Fp);
}

bool CGnuplot::Check()
{
	if (Fp == NULL) {
		return false;
	}
	return true;
}

void CGnuplot::Command(const char* format, ...)
{
	char buf[1024];
	va_list ap;

	va_start(ap, format);
	vsprintf_s(buf, format, ap);
	va_end(ap);

	fprintf(Fp, "%s\n", buf);
	Flush();
}

void CGnuplot::DrawFunc(const char* format)
{
	Command("plot %s", format);
}

void CGnuplot::SetXLabel(const char* format)
{
	Command("set xlabel '%s'", format);
}

void CGnuplot::SetYLabel(const char* format)
{
	Command("set ylabel '%s'", format);
}

void CGnuplot::SetXRange(const double x_min, const double x_max)
{
	Command("set xrange [%f:%f]", x_min, x_max);
}

void CGnuplot::SetYRange(const double y_min, const double y_max)
{
	Command("set yrange [%f:%f]", y_min, y_max);
}

void CGnuplot::Flush()
{
	fflush(Fp);
}

void CGnuplot::Replot()
{
	Command("replot");
}