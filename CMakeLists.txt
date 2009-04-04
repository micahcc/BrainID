cmake_minimum_required(VERSION 2.6)
SET(CMAKE_VERBOSE_MAKEFILE 1)

PROJECT(brainid)

# Find ITK.
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

# Find dysii
#FIND_PACKAGE(dysii)
#if(dysii_found)
#    include(${dysii_use_file})
#else(itk_found)
#    MESSAGE(FATAL_ERROR "Cannont build without dysii, please set dysii_DIR.")
#endif(itk_found)

find_path(dysii_INCLUDE_DIRS indii/ml/filter/ParticleFilter.hpp)
FIND_LIBRARY(dysii_LIBRARY_DIRS NAMES dysii PATH /usr/lib) 

find_path(BOOST_INCLUDE_DIRS boost/mpi.h)
FIND_LIBRARY(BOOST_LIBRARY_DIRS NAMES boost_serialization PATH /usr/lib) 
#find_path(MPI_INCLUDE mpi.hpp)

SUBDIRS(code)
SUBDIRS(testing)
