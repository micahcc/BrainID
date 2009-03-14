cmake_minimum_required(VERSION 2.6)

# Find ITK.
#FIND_PACKAGE(ITK)
#IF(ITK_FOUND)
#  INCLUDE(${ITK_USE_FILE})
#ELSE(ITK_FOUND)
#  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
#ENDIF(ITK_FOUND)

include_directories(${brainid_SOURCE_DIR}/code/include ${DYSII_INCLUDE})
link_directories(${brainid_SOURCE_DIR}/code)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_executable(particle_test particle_test.cpp)

ADD_EXECUTABLE(segment_alltogether segment_alltogether.cpp)
ADD_EXECUTABLE(segment_sections segment_sections.cpp)
ADD_EXECUTABLE(segment_times segment_times.cpp)

TARGET_LINK_LIBRARIES(particle_test mpi boost_mpi-mt ITKIO ITKCommon dysii gsl
    lapack)

TARGET_LINK_LIBRARIES(segment_alltogether ITKCommon segment ITKIO)
TARGET_LINK_LIBRARIES(segment_sections ITKCommon segment ITKIO)
TARGET_LINK_LIBRARIES(segment_times segment ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO)
