set terminal postscript eps enhanced colour solid 24
set output "images/test10_tree.eps"

set size 2.0,2.0
set palette rgbformulae 30,31,32

set cbrange [0:0.25]

plot "results/test10_tree.out" using 1:2:3 notitle with image

