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
    ${dysii_INCLUDE_DIRS})
LINK_DIRECTORIES(${brainid_SOURCE_DIR}/code ${BOOST_LIBRARY_DIRS} 
    ${dysii_LIBRARY_DIRS})

ADD_LIBRARY(segment segment.cpp)
ADD_LIBRARY(BoldModel BoldModel.cpp)

#TARGET_LINK_LIBRARIES(bold particle ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)

ADD_EXECUTABLE(unbias unbias.cpp)
ADD_EXECUTABLE(brainid brainid.cpp)
#ADD_EXECUTABLE(brainid brainid.cpp StratifiedParticleResampler.cpp)
ADD_EXECUTABLE(fmri_timeseries fmri_timeseries.cpp)
ADD_EXECUTABLE(boldgen boldgen.cpp)

TARGET_LINK_LIBRARIES(fmri_timeseries segment ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(unbias ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(brainid boost_serialization-gcc41-mt boost_mpi-mt BoldModel ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO mpi gsl lapack dysii)
TARGET_LINK_LIBRARIES(boldgen boost_serialization-gcc41-mt boost_mpi-mt BoldModel ITKAlgorithms ITKBasicFilters 
    ITKCommon ITKIO mpi gsl lapack dysii)
