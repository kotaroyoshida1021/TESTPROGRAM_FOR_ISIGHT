#include "pch.h"
#include <iostream>
#include "Obj_Coordinates.h"
#include "IntervalReduction.h"

dbl BisectionMethod::Solution(dbl(*func)(dbl s), dbl INIT, dbl END) {
	init = INIT;
	end = END;
	int	k;
	dbl	ya, yb, ret, xp, xm, xc, yc;

	ya = func(init);
	yb = func(end);
	ret = 0.0;

	if (fabs(ya) <= eps) {
		ret = init;
		return ret;
	}
	if (fabs(yb) <= eps) {
		ret = end;
		return ret;
	}

	if (ya > 0.00 && yb < 0.00) {
		xp = init;
		xm = end;
	}
	else if (ya < 0.00 && yb > 0.00) {
		xm = init;
		xp = end;
	}
	else {
#if 1
		fprintf(stderr, "interval [ %lf  %lf ]\n", init, end);
		fprintf(stderr, "%lf and %lf have same sign\n", ya, yb);
#endif
		return end;
	}

	for (k = 0; k < limit_Num; k++) {
#if Debug
		fprintf(stderr, "interval [ %lf , %lf ]\n", xp, xm);
#endif
		xc = (xp + xm) / 2.00;
		yc = func(xc);
		if (fabs(yc) <= eps) {
			ret = xc;
			return ret;
		}
		if (fabs(xp - xm) <= eps) {
			ret = xc;
			return ret;
		}
		if (yc > 0.00) {
			xp = xc;
		}
		else {
			xm = xc;
		}
	}

	fprintf(stderr, "IntervalReduction timeout\n");
	return ret;
}