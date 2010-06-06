cmake_minimum_required(VERSION 2.6)
SET(CMAKE_VERBOSE_MAKEFILE)
INCLUDE (CheckIncludeFiles)

PROJECT(dysii)

CHECK_INCLUDE_FILES(math.h HAVE_MATH_H)

IF(NOT HAVE_MATH_H)
    MESSAGE(FATAL_ERROR "You need to have math.h")
ENDIF()

find_path(GSL include/gsl lib/libgsl.a)
find_path(MPI include/mpi.h lib/libmpi.la)
find_path(BOOST include/boost/numeric/ublas/vector.h
    include/boost/numeric/ublas/matrix.hpp 
    include/boost/numeric/bindings/traits/ublas_vector.hpp
    include/boost/numeric/bindings/traits/ublas_matrix.hpp
    include/boost/serialization/serialization.hpp lib/boost/)

AUX_SOURCE_DIRECTORY(src/indii/ml/aux SOURCES )
AUX_SOURCE_DIRECTORY(src/indii/ml/data SOURCES )
AUX_SOURCE_DIRECTORY(src/indii/ml/filter SOURCES )
AUX_SOURCE_DIRECTORY(src/indii/ml/ode SOURCES )
AUX_SOURCE_DIRECTORY(src/indii/ml/sde SOURCES )

INCLUDE_DIRECTORIES(${GSL}/include ${MPI}/include ${BOOST}/include)

ADD_LIBRARY(dysii STATIC ${SOURCES})

INSTALL(TARGETS dysii LIBRARY ARCHIVE DESTINATION lib/)

INSTALL(DIRECTORY src/indii DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

SET(dysii_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(dysii_LIBRARY_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
