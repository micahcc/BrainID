cmake_minimum_required(VERSION 2.6)
SET(CMAKE_VERBOSE_MAKEFILE)

PROJECT(brainid)

# Find ITK.
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

SUBDIRS(code)
SUBDIRS(testing)
