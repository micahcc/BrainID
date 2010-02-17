cmake_minimum_required(VERSION 2.6)
SET(CMAKE_VERBOSE_MAKEFILE 0)

PROJECT(brainid)

# Find ITK.
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -pg")

set(CMAKE_CXX_FLAGS_GENERIC "")

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

find_path(MPI_INCLUDE_DIRS mpi.hpp)
FIND_LIBRARY(MPI_LIBRARY_DIRS NAMES mpi PATH /usr/lib) 

find_path(GSL_INCLUDE_DIRS gsl/gsl_errno.h)
FIND_LIBRARY(GSL_LIBRARY_DIRS NAMES gsl PATH /usr/lib) 

#if(EXISTS ${brainid_SOURCE_DIR}/.git)
#    set(ARGS svn info)
#    execute_process( COMMAND git ${ARGS} OUTPUT_VARIABLE VERSION)
#else(EXISTS ${brainid_SOURCE_DIR}/.git)
#    set(ARGS info)
#    execute_process( COMMAND svn ${ARGS} OUTPUT_VARIABLE VERSION)
#endif(EXISTS ${brainid_SOURCE_DIR}/.git)

#string(REGEX MATCH "Revision: [0-9]*" VERSION2 ${VERSION})
#string(REGEX MATCH "[0-9]+" VERSION ${VERSION2})
#set(VERSION "#define BRAINID_VERSION \"${VERSION}\"\n")
#message("Writing version.h: ${VERSION}")

#file(WRITE ${brainid_SOURCE_DIR}/code/include/version.h "${VERSION}")
#
#add_definitions(-DVERSION="${VERSION}")

SUBDIRS(code/utils code testing)
