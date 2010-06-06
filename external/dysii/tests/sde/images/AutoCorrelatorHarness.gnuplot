set terminal postscript eps enhanced colour solid 24
set output "images/AutoCorrelatorHarness.eps"

set grid
set size 2,2

set style line 2 linetype 1

plot "results/AutoCorrelatorHarness.out" using 1:2 with lines linestyle 2 notitle

