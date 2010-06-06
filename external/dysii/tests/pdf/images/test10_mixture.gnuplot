set terminal postscript eps enhanced colour solid 24
set output "images/test10_mixture.eps"

set size 2.0,2.0
unset grid
set palette rgbformulae 30,31,32

set cbrange [0:0.25]

plot "results/test10_mixture.out" using 1:2:3 notitle with image

