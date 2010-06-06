set terminal postscript eps enhanced colour solid 24
set output "images/ParticleFilterHarness.eps"

set grid
set size 2,2
set style line 1 linetype 1 linewidth 1 pointtype 2 pointsize 2
set style line 2 linetype 2 linewidth 1 pointtype 2 pointsize 2
set style line 3 linetype 3 linewidth 1 pointtype 2 pointsize 2
set style line 4 linetype 4 linewidth 8 pointtype 2 pointsize 2

plot \
"results/ParticleFilterHarness_actual.out" using 2:3 title "actual" with lines linestyle 1,\
"results/ParticleFilterHarness_actual.out" using (floor($0)%10==0 ? $2 : 1/0):3 notitle with points linestyle 1,\
"results/ParticleFilterHarness_filter.out" using 2:3 title "filtered" with lines linestyle 2, \
"results/ParticleFilterHarness_filter.out" using (floor($0)%10==0 ? $2 : 1/0):3:(2*sqrt($4)):(2*sqrt($7)) notitle with xyerrorbars linestyle 2, \
0 title "wall" with lines linestyle 4

