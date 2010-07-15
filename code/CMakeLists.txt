cmake_minimum_required(VERSION 2.6)
SET(CMAKE_VERBOSE_MAKEFILE 1)

# Find ITK.
#FIND_PACKAGE(ITK)
#IF(ITK_FOUND)
#  INCLUDE(${ITK_USE_FILE})
#ELSE(ITK_FOUND)
#  MESSAGE(FATAL_ERROR "Cannot build without ITK. Please set ITK_DIR.")
#ENDIF(ITK_FOUND)

INCLUDE_DIRECTORIES(
${brainid_SOURCE_DIR}/code/include 
${BOOST_INCLUDE_DIRS}
${MPI_INCLUDE_DIRS} 
${GSL_INCLUDE_DIRS}
${brainid_SOURCE_DIR}/code/utils
${dysii_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
    ${brainid_SOURCE_DIR}/code/utils/vul
    ${brainid_SOURCE_DIR}/code 
    ${BOOST_LIBRARY_DIRS} 
    ${MPI_LIBRARY_DIRS} 
    ${GSL_LIBRARY_DIRS}
    ${dysii_LIBRARY_DIRS}
)

ADD_LIBRARY(modNiftiImageIO modNiftiImageIO.cxx)
TARGET_LINK_LIBRARIES(modNiftiImageIO ITKIO)

ADD_LIBRARY(segment segment.cpp)
TARGET_LINK_LIBRARIES(segment ITKAlgorithms ITKBasicFilters ITKCommon gsl gslcblas tools boost_mpi mpi)

ADD_LIBRARY(BoldModel BoldModel.cpp)

ADD_LIBRARY(tools tools.cc)
TARGET_LINK_LIBRARIES(tools ITKAlgorithms ITKBasicFilters ITKCommon tools gsl gslcblas )

#TARGET_LINK_LIBRARIES(bold particle ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)

ADD_EXECUTABLE(unbias unbias.cpp)
ADD_EXECUTABLE(brainid brainid.cpp)
ADD_EXECUTABLE(parammap_calc calc_parammap.cpp)
ADD_EXECUTABLE(printtags printtags.cpp)
ADD_EXECUTABLE(setorientation setorientation.cpp)
#ADD_EXECUTABLE(fmri_timeseries fmri_timeseries.cpp)
ADD_EXECUTABLE(chop chop.cpp)
ADD_EXECUTABLE(boldgen boldgen.cpp)
ADD_EXECUTABLE(concatinate concat.cpp)
ADD_EXECUTABLE(prune prune.cpp)
ADD_EXECUTABLE(readdicom readdicom.cpp)
ADD_EXECUTABLE(applymask applymask.cpp)
ADD_EXECUTABLE(reorientby reorientby.cpp)
ADD_EXECUTABLE(makeest makeest.cpp)
ADD_EXECUTABLE(fft fft.cpp)
ADD_EXECUTABLE(parammap_preprocess preprocess.cpp)
ADD_EXECUTABLE(parammap_generate make_param_image.cpp)
ADD_EXECUTABLE(parammap_eval parammap_eval.cpp)
#ADD_EXECUTABLE(parammap_evaldiff parammap_diff.cpp)
ADD_EXECUTABLE(mse mse.cpp)
ADD_EXECUTABLE(extract_pos extract_pos.cc)

TARGET_LINK_LIBRARIES(makeest modNiftiImageIO ITKIO)
TARGET_LINK_LIBRARIES(prune modNiftiImageIO ITKIO)
TARGET_LINK_LIBRARIES(chop ITKBasicFilters ITKIO vullib)
TARGET_LINK_LIBRARIES(concatinate modNiftiImageIO ITKIO)
#TARGET_LINK_LIBRARIES(fmri_timeseries modNiftiImageIO segment 
#    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO 
#    vullib)
TARGET_LINK_LIBRARIES(unbias ITKAlgorithms ITKBasicFilters ITKCommon ITKIO )
TARGET_LINK_LIBRARIES(printtags modNiftiImageIO ITKIO)
TARGET_LINK_LIBRARIES(setorientation modNiftiImageIO ITKIO)
TARGET_LINK_LIBRARIES(brainid tools gfortran modNiftiImageIO blas lapack mpi 
    boost_serialization boost_mpi BoldModel 
    ITKIO mpi gsl gslcblas dysii vullib)
TARGET_LINK_LIBRARIES(parammap_preprocess segment tools modNiftiImageIO 
    ITKAlgorithms ITKIO gsl gslcblas vullib )
TARGET_LINK_LIBRARIES(parammap_calc segment tools gfortran modNiftiImageIO blas lapack mpi 
    boost_serialization boost_mpi BoldModel ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO gsl gslcblas dysii vullib )
TARGET_LINK_LIBRARIES(parammap_eval modNiftiImageIO ITKIO ITKCommon vullib tools)
#TARGET_LINK_LIBRARIES(parammap_evaldiff modNiftiImageIO ITKIO ITKCommon vullib tools)
TARGET_LINK_LIBRARIES(mse modNiftiImageIO ITKIO ITKCommon vullib tools)
TARGET_LINK_LIBRARIES(boldgen tools gfortran blas lapack modNiftiImageIO boost_serialization boost_mpi
    BoldModel ITKAlgorithms ITKBasicFilters 
    ITKCommon ITKIO gsl gslcblas mpi dysii vullib)
TARGET_LINK_LIBRARIES(fft tools modNiftiImageIO ITKIO vullib)
TARGET_LINK_LIBRARIES(extract_pos modNiftiImageIO ITKIO ITKCommon vullib)

TARGET_LINK_LIBRARIES(readdicom modNiftiImageIO segment ITKAlgorithms ITKBasicFilters ITKCommon ITKIO vullib)
TARGET_LINK_LIBRARIES(applymask modNiftiImageIO segment ITKAlgorithms ITKBasicFilters ITKCommon ITKIO vullib)
TARGET_LINK_LIBRARIES(reorientby tools ITKCommon ITKIO )
TARGET_LINK_LIBRARIES(parammap_generate gsl modNiftiImageIO segment tools ITKBasicFilters ITKCommon ITKIO 
    vullib)
