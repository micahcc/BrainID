set terminal postscript eps enhanced colour solid 24
set output "images/EquilibriumSamplerHarness.eps"

set size 2,2

plot "results/EquilibriumSamplerHarness.out" using 1:2 with impulses notitle

