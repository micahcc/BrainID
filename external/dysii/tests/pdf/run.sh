#!/bin/bash

# Create directory for results
mkdir -p results

# Run all tests
for i in `seq 1 2; seq 5 6;`
do
  echo "Test $i..."
  ./test$i > results/test$i.out
done

for i in `seq 4 4`
do
  echo "Test $i..."
  ./test$i
  gnuplot images/test$i.gnuplot
  convert images/test$i.eps images/test$i.png
done

for i in `seq 3 3; seq 7 8; seq 11 12`
do
  echo "Test $i..."
  mpirun -np 4 ./test$i > results/test$i.out
done

for i in `seq 10 10`
do
  echo "Test $i..."
  ./test$i > results/test$i.out
  gnuplot images/test${i}_mixture.gnuplot
  convert images/test${i}_mixture.eps images/test${i}_mixture.png
  gnuplot images/test${i}_tree.gnuplot
  convert images/test${i}_tree.eps images/test${i}_tree.png
done

