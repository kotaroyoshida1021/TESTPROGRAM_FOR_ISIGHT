set xlabel offset 0,0
set ylabel offset 0,0

set size ratio -1
plot "shape.txt" using 1:2 with lines linewidth 1.5 title "opt_shape", "wire.txt" using 1:2 with lines linewidth 1.5 title "wire_shape"
set terminal pngcairo crop
set output "AlignWire_xyview.png"

replot

reset


set xlabel offset 0,0
set ylabel offset 0,0

set size ratio -1
plot "shape.txt" using 1:3 with lines linewidth 1.5 title "opt_shape", "wire.txt" using 1:3 with lines linewidth 1.5 title "wire_shape"
set terminal pngcairo crop
set output "AlignWire_xzview.png"

replot

reset


set xlabel offset 0,0
set ylabel offset 0,0

set size ratio -1
plot "functions.txt" using 1:2 with lines linewidth 1.5 title "beta"
set terminal pngcairo crop
set output "beta.png"

replot

reset

set xlabel offset 0,0
set ylabel offset 0,0

set size ratio -1
plot "functions.txt" using 1:3 with lines linewidth 1.5 title "omg_eta"
set terminal pngcairo crop
set output "omgEta.png"

replot

reset