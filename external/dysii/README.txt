------------------------------------------------------------------------
dysii Dynamic Systems Library
  http://www.indii.org/software/dysii/

Lawrence Murray
  lawrence@indii.org
------------------------------------------------------------------------

1. Introduction
2. Installation
3. Documentation
4. Examples
5. Tests
6. Further information


1. Introduction
------------------------------------------------------------------------

This C++ library provides a collection of classes useful for large-scale
probabilistic inference and learning within dynamical systems. It features the
following:

  Filtering and smoothing:

    * Kalman filter and smoother
    * Rauch-Tung-Striebel (RTS) smoother
    * Unscented Kalman filter and smoother
    * Particle filter and forward-backward smoother
    * Kernel forward-backward and two-filter smoothers
    * Multiple resampling strategies for particle filters, including
      stratified, auxiliary and regularised resampling

  Probability distributions:

    * Gaussian and Gaussian mixture distributions
    * Dirac and Dirac mixture (weighted sample set) distributions
    * Kernel density estimators

  Differential equations:
 
    * Adaptive numerical solvers for ordinary and stochastic differential
      equations, including Euler-Maruyama and stochastic Runge-Kutta.
    * Autocorrelator and equilibrium distribution sampler.

  Parallelisation using MPI:
  
    * Parallel particle filter and smoother
    * Parallel kernel forward-backward and two-filter smoother
    * Distributed storage of mixture densities
    * Distibuted kd tree evaluations, including dual- and self-tree
      evaluations.

  Data management:

    * Serialization of vectors, matrices and probability distributions for
      fast and convenient data management, using Boost.Serialization.
    * Text file reader and writer.

  Performance:

    * Use of BLAS and LAPACK
    * Template meta-programming
    * Code profiling
    * Compiler optimisation

The library has been optimised for performance, while maintaining a modularity
and generality that makes it suitable for a wide range of applications.


2. Installation
------------------------------------------------------------------------

See the included INSTALL.txt file.


3. Documentation
------------------------------------------------------------------------

Documentation may be built using:

doxygen

This will create a docs/ directory containing the documentation. Open
docs/html/index.html in your browser for the main page. Alternatively,
documentation for the latest version of the library is available online at
<http://www.indii.org/software/dysii/>.


4. Examples
------------------------------------------------------------------------

Example applications using dysii are available from the website at
<http://www.indii.org/software/dysii/download/>.


5. Tests
------------------------------------------------------------------------

A test suite is provided in the tests/ directory. See the README.txt file in
this directory for more information. Test results are also available from
the website at <http://www.indii.org/software/dysii/documentation/>.

The tests may also serve as useful examples of how to use the library.


6. Further information
------------------------------------------------------------------------

For further information see the website <http://www.indii.org/software/dysii/>
or contact the author, Lawrence Murray <lawrence@indii.org>.

------------------------------------------------------------------------
