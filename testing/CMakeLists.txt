cmake_minimum_required(VERSION 2.6)
SET(CMAKE_VERBOSE_MAKEFILE 1)


# Find ITK.
#FIND_PACKAGE(ITK)
#IF(ITK_FOUND)
#  INCLUDE(${ITK_USE_FILE})
#ELSE(ITK_FOUND)
#  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
#ENDIF(ITK_FOUND)

INCLUDE_DIRECTORIES(${brainid_SOURCE_DIR}/code/include ${BOOST_INCLUDE_DIRS}
    ${dysii_INCLUDE_DIRS} ${MPI_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS})
LINK_DIRECTORIES(${brainid_SOURCE_DIR}/code ${BOOST_LIBRARY_DIRS} 
    ${dysii_LIBRARY_DIRS} ${MPI_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS})

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_executable(particle_test particle_test.cpp)

ADD_EXECUTABLE(segment_alltogether segment_alltogether.cpp)
ADD_EXECUTABLE(segment_sections segment_sections.cpp)
ADD_EXECUTABLE(segment_times segment_times.cpp)

TARGET_LINK_LIBRARIES(particle_test mpi boost_serialization boost_mpi BoldModel ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO mpi gslcblas gsl gfortran blas lapack dysii)

TARGET_LINK_LIBRARIES(segment_alltogether ITKCommon segment ITKIO
    ITKAlgorithms)
TARGET_LINK_LIBRARIES(segment_sections ITKCommon segment ITKIO ITKAlgorithms)
TARGET_LINK_LIBRARIES(segment_times segment ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO)
