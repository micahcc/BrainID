set terminal postscript eps enhanced colour solid 24
set output "images/KalmanSmootherHarness.eps"

set grid
set size 2,2
set style line 1 linetype 1 linewidth 1 pointtype 2 pointsize 2
set style line 2 linetype 2 linewidth 1 pointtype 2 pointsize 2
set style line 3 linetype 3 linewidth 1 pointtype 2 pointsize 2
set style line 4 linetype 4 linewidth 8 pointtype 2 pointsize 2
set style line 5 linetype 4 linewidth 1 pointtype 2 pointsize 2

plot \
"results/KalmanSmootherHarness_actual.out" using 2:3 title "actual" with lines linestyle 1,\
"results/KalmanSmootherHarness_actual.out" using (floor($0)%10==0 ? $2 : 1/0):3 notitle with points linestyle 1,\
"results/KalmanSmootherHarness_filter.out" using 2:3 title "filtered" with lines linestyle 2, \
"results/KalmanSmootherHarness_filter.out" using (floor($0)%10==0 ? $2 : 1/0):3:7:13 notitle with xyerrorbars linestyle 2, \
"results/KalmanSmootherHarness_backfilter.out" using 2:3 title "backward filtered" with lines linestyle 5, \
"results/KalmanSmootherHarness_backfilter.out" using (floor($0)%10==0 ? $2 : 1/0):3:7:13 notitle with xyerrorbars linestyle 5, \
"results/KalmanSmootherHarness_smooth.out" using 2:3 title "smoothed" with lines linestyle 3, \
"results/KalmanSmootherHarness_smooth.out" using (floor($0)%10==0 ? $2 : 1/0):3:7:13 notitle with xyerrorbars linestyle 3, \
0 title "wall" with lines linestyle 4
