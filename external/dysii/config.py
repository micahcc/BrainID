#
# Configuration file.
#
# Any values given in this file override the default build options on
# your system, but are overriden by options given on the command
# line. To use the default value for a particular option, simply
# comment it out. Options may also be specified on the command line
# when calling scons.
#
# To see the full list of options, type `scons -h`.
#

# gcc compiler specifics
CXX = 'g++'
CXXFLAGS = '-Wall -O3 -funroll-loops'

# Intel compiler specifics
#CXX = 'icpc'
#CXXFLAGS = '-O3 -funroll-loops -wd981'

# OpenMPI specifics
CXXFLAGS += ' `mpic++ -showme:compile`'

