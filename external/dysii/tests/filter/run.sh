#!/bin/bash

# Create directory for results
mkdir -p results

# Run all tests
echo Unscented transformation... 
./UnscentedTransformationHarness
echo

echo Kalman filter...
./KalmanFilterHarness
gnuplot images/KalmanFilterHarness.gnuplot
echo
echo
convert images/KalmanFilterHarness.eps images/KalmanFilterHarness.png

echo Kalman smoother... 
./KalmanSmootherHarness
echo
echo
gnuplot images/KalmanSmootherHarness.gnuplot
convert images/KalmanSmootherHarness.eps images/KalmanSmootherHarness.png

echo Rauch-Tung-Striebel smoother...
./RauchTungStriebelHarness
echo
echo
gnuplot images/RauchTungStriebelHarness.gnuplot
convert images/RauchTungStriebelHarness.eps images/RauchTungStriebelHarness.png

echo Unscented Kalman filter... 
./UnscentedKalmanFilterHarness
echo
echo
gnuplot images/UnscentedKalmanFilterHarness.gnuplot
convert images/UnscentedKalmanFilterHarness.eps images/UnscentedKalmanFilterHarness.png

echo Unscented Kalman smoother... 
./UnscentedKalmanSmootherHarness
echo
echo
gnuplot images/UnscentedKalmanSmootherHarness.gnuplot
convert images/UnscentedKalmanSmootherHarness.eps images/UnscentedKalmanSmootherHarness.png

echo Particle filter... 
mpirun -np 4 ./ParticleFilterHarness
echo
echo
gnuplot images/ParticleFilterHarness.gnuplot
convert images/ParticleFilterHarness.eps images/ParticleFilterHarness.png

echo Particle smoother... 
mpirun -np 4 ./ParticleSmootherHarness
echo
echo
gnuplot images/ParticleSmootherHarness.gnuplot
convert images/ParticleSmootherHarness.eps images/ParticleSmootherHarness.png

echo Kernel forward-backward smoother... 
mpirun -np 4 ./KernelForwardBackwardSmootherHarness
echo
echo
gnuplot images/KernelForwardBackwardSmootherHarness.gnuplot
convert images/KernelForwardBackwardSmootherHarness.eps images/KernelForwardBackwardSmootherHarness.png

echo Kernel two-filter smoother... 
mpirun -np 4 ./KernelTwoFilterSmootherHarness
echo
echo
gnuplot images/KernelTwoFilterSmootherHarness.gnuplot
convert images/KernelTwoFilterSmootherHarness.eps images/KernelTwoFilterSmootherHarness.png

