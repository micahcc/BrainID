------------------------------------------------------------------------
dysii Dynamic Systems Library Test Suite
  http://www.indii.org/software/dysii/

Lawrence Murray
  lawrence@indii.org
------------------------------------------------------------------------

Test suite for the dysii Dynamic Systems Library.


1. Requirements
------------------------------------------------------------------------

The dysii library must have been successfully installed in order to run
the test suite. In addition, the following are required:

  * Scons <http://www.scons.org/>
  * Doxygen <http://www.doxygen.org/>
  * Gnuplot <http://www.gnuplot.info/>
  * ImageMagick <http://www.imagemagick.org/>

All of these are commonly installed with Linux distributions or
readily available through package managers.


2. Usage
------------------------------------------------------------------------

Each subdirectory contains a set of tests for a particular component
of the library. To run a set of tests, change the working directory to
the relevant subdirectory. Compile the tests using:

scons

Run the tests using:

./run.sh

And compile the results using:

doxygen

This will create a docs/ directory containing the test results. Open
docs/html/index.html in your browser for the main page. Access the
Files page and view each test to see the results.

------------------------------------------------------------------------
