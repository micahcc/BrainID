cmake_minimum_required(VERSION 2.6)
#SET(CMAKE_VERBOSE_MAKEFILE 1)

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

if(EXISTS ${brainid_SOURCE_DIR}/.git)
    set(ARGS log -n 1 --format=%H)
    execute_process( COMMAND git ${ARGS} OUTPUT_VARIABLE VERSION)
    string(REGEX MATCH "[0-9A-Za-z]*" VERSION ${VERSION})
else(EXISTS ${brainid_SOURCE_DIR}/.git)
    set(ARGS info)
    execute_process( COMMAND svn ${ARGS} OUTPUT_VARIABLE VERSION)
    string(REGEX MATCH "Revision: [0-9]*" VERSION2 ${VERSION})
    string(REGEX MATCH "[0-9]+" VERSION ${VERSION2})
endif(EXISTS ${brainid_SOURCE_DIR}/.git)

execute_process(COMMAND date +D%m_%d_%y_T%H_%M OUTPUT_VARIABLE TODAY)
string(REGEX MATCH "[0-9A-Za-z_]*" TODAY ${TODAY})

add_definitions(-DBRAINID_VERSION="${VERSION}_${TODAY}")
message("${VERSION}_${TODAY}")

SUBDIRS(code/utils code testing)
