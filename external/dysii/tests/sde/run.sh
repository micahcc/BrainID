#!/bin/bash

# Create directory for results
mkdir -p results

# Run all tests

./StochasticAdaptiveRungeKuttaHarness
gnuplot images/StochasticAdaptiveRungeKuttaHarness.gnuplot
convert images/StochasticAdaptiveRungeKuttaHarness.eps images/StochasticAdaptiveRungeKuttaHarness.png

./StochasticAdaptiveEulerMaruyamaHarness
gnuplot images/StochasticAdaptiveEulerMaruyamaHarness.gnuplot
convert images/StochasticAdaptiveEulerMaruyamaHarness.eps images/StochasticAdaptiveEulerMaruyamaHarness.png

./AutoCorrelatorHarness
gnuplot images/AutoCorrelatorHarness.gnuplot
convert images/AutoCorrelatorHarness.eps images/AutoCorrelatorHarness.png

./EquilibriumSamplerHarness
gnuplot images/EquilibriumSamplerHarness.gnuplot
convert images/EquilibriumSamplerHarness.eps images/EquilibriumSamplerHarness.png

