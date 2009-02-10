cmake_minimum_required(VERSION 2.6)

# Find ITK.
#FIND_PACKAGE(ITK)
#IF(ITK_FOUND)
#  INCLUDE(${ITK_USE_FILE})
#ELSE(ITK_FOUND)
#  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
#ENDIF(ITK_FOUND)

include_directories(${brainid_SOURCE_DIR}/code/include)
link_directories(${brainid_SOURCE_DIR}/code)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


ADD_EXECUTABLE(bold_test bold_test.cpp)
ADD_EXECUTABLE(particle_test particle_test.cpp)

ADD_EXECUTABLE(sumfmri_test_alltogether sumfmri_testalltogether.cpp)
ADD_EXECUTABLE(sumfmri_test_sections sumfmri_testsections.cpp)
ADD_EXECUTABLE(sumfmri_test_times sumfmri_testtimes.cpp)


TARGET_LINK_LIBRARIES(bold_test bold particle ITKCommon)
TARGET_LINK_LIBRARIES(sumfmri_test_alltogether ITKCommon sumfmri)
TARGET_LINK_LIBRARIES(sumfmri_test_sections ITKCommon sumfmri)
TARGET_LINK_LIBRARIES(sumfmri_test_times ITKCommon sumfmri)
