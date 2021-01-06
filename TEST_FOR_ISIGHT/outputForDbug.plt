set xlabel offset 0,0
set ylabel offset 0,0
set datafile separator ","
plot "EvalFunc_and_Conds.csv" using 1:2 with points linewidth 1.5 title "OBJECTIVE"
set terminal pngcairo crop
set output "OBJ.png"

replot

reset


set xlabel offset 0,0
set ylabel offset 0,0
set datafile separator ","

#set size ratio -1
plot "EvalFunc_and_Conds.csv" using 1:3 with points linewidth 1.5 title "COND1","EvalFunc_and_Conds.csv" using 1:4 with points linewidth 1.5 title "COND2","EvalFunc_and_Conds.csv" using 1:5 with points linewidth 1.5 title "COND3"
set terminal pngcairo crop
set output "COND1.png"

replot

reset


set xlabel offset 0,0
set ylabel offset 0,0
set datafile separator ","

#set size ratio -1
plot "EvalFunc_and_Conds.csv" using 1:6 with points linewidth 1.5 title "COND1","EvalFunc_and_Conds.csv" using 1:7 with points linewidth 1.5 title "COND2","EvalFunc_and_Conds.csv" using 1:8 with points linewidth 1.5 title "COND3"
set terminal pngcairo crop
set output "COND2.png"

replot

reset