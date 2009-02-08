cmake_minimum_required(VERSION 2.6)

# Find ITK.
#FIND_PACKAGE(ITK)
#IF(ITK_FOUND)
#  INCLUDE(${ITK_USE_FILE})
#ELSE(ITK_FOUND)
#  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
#ENDIF(ITK_FOUND)

INCLUDE_DIRECTORIES(${brainid_SOURCE_DIR}/code/include)
LINK_DIRECTORIES(${brainid_SOURCE_DIR}/code)

ADD_LIBRARY(bold bold.cpp)
ADD_LIBRARY(particle particle.cpp)
