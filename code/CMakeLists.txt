cmake_minimum_required(VERSION 2.6)

# Find ITK.
#FIND_PACKAGE(ITK)
#IF(ITK_FOUND)
#  INCLUDE(${ITK_USE_FILE})
#ELSE(ITK_FOUND)
#  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
#ENDIF(ITK_FOUND)

INCLUDE_DIRECTORIES(${brainid_SOURCE_DIR}/code/include ${SMCTC_INCLUDE})
LINK_DIRECTORIES(${brainid_SOURCE_DIR}/code)

ADD_LIBRARY(bold bold.cpp)
ADD_LIBRARY(particle particle.cpp)
ADD_LIBRARY(segment segment.cpp)

#TARGET_LINK_LIBRARIES(bold particle ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)

ADD_EXECUTABLE(unbias unbias.cpp)
ADD_EXECUTABLE(brainid brainid.cpp)
ADD_EXECUTABLE(fmri_timeseries fmri_timeseries.cpp)

TARGET_LINK_LIBRARIES(fmri_timeseries segment ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(unbias ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(brainid bold particle ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO mpi boost_mpi-mt dysii gsl lapack)
