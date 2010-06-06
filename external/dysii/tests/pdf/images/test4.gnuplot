set terminal postscript eps enhanced colour solid 24
set output "images/test4.eps"

set grid
set size 2,2

set style line 2 linetype 1 linewidth 10
set style line 3 linetype 1 linewidth 3
set style line 4 linetype 2 linewidth 10
set style line 5 linetype 2 linewidth 3
set style line 6 linetype 3 linewidth 10
set style line 7 linetype 3 linewidth 3

plot "results/test4_actual.out" using 1:2 with lines linestyle 2 title "actual mean",\
"results/test4_actual.out" using 1:($2+sqrt($3)) with lines linestyle 3 title "actual std. dev.",\
"results/test4_actual.out" using 1:($2-sqrt($3)) with lines linestyle 3 notitle,\
"results/test4_expected.out" using 1:2 with lines linestyle 4 title "expected mean",\
"results/test4_expected.out" using 1:($2+sqrt($3)) with lines linestyle 5 title "expected std. dev.",\
"results/test4_expected.out" using 1:($2-sqrt($3)) with lines linestyle 5 notitle,\
"results/test4_sample.out" using 1:2 with lines linestyle 6 title "importance sampled mean",\
"results/test4_sample.out" using 1:($2+sqrt($3)) with lines linestyle 7 title "importance sampled std. dev.",\
"results/test4_sample.out" using 1:($2-sqrt($3)) with lines linestyle 7 notitle
