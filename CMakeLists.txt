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

find_path(SMCTC_INCLUDE history.hh moveset.hh  particle.hh rng.hh smc-exception.hh smctc.hh)
find_library(SMCTC_LIBRARY libsmctc.a)

SUBDIRS(code)
SUBDIRS(testing)
