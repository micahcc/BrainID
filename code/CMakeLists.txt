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
${dysii_INCLUDE_DIRS} 
${MPI_INCLUDE_DIRS} 
${GSL_INCLUDE_DIRS}
${brainid_SOURCE_DIR}/code/utils
)

LINK_DIRECTORIES(
    ${brainid_SOURCE_DIR}/code/utils/vul
    ${brainid_SOURCE_DIR}/code 
    ${BOOST_LIBRARY_DIRS} 
    ${dysii_LIBRARY_DIRS} 
    ${MPI_LIBRARY_DIRS} 
    ${GSL_LIBRARY_DIRS}
)

ADD_LIBRARY(modNiftiImageIO modNiftiImageIO.cxx)
ADD_LIBRARY(segment segment.cpp)
ADD_LIBRARY(activate activate.cpp)
ADD_LIBRARY(BoldModel BoldModel.cpp)
ADD_LIBRARY(tools tools.cc)

#TARGET_LINK_LIBRARIES(bold particle ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)

ADD_EXECUTABLE(unbias unbias.cpp)
ADD_EXECUTABLE(brainid brainid.cpp)
ADD_EXECUTABLE(calc_parammap calc_parammap.cpp)
ADD_EXECUTABLE(printtags printtags.cpp)
ADD_EXECUTABLE(copyorigin copy.cpp)
ADD_EXECUTABLE(fmri_timeseries fmri_timeseries.cpp)
ADD_EXECUTABLE(boldgen boldgen.cpp)
ADD_EXECUTABLE(concatinate concat.cpp)
ADD_EXECUTABLE(prune prune.cpp)
ADD_EXECUTABLE(readdicom readdicom.cpp)
ADD_EXECUTABLE(makespline makespline.cpp)
ADD_EXECUTABLE(applymask applymask.cpp)
ADD_EXECUTABLE(fft fft.cpp)
ADD_EXECUTABLE(makesine makesine.cpp)
ADD_EXECUTABLE(actmap activate_map.cpp)
ADD_EXECUTABLE(mean_filter mean_filter.cpp)
ADD_EXECUTABLE(make_param_image make_param_image.cpp)

TARGET_LINK_LIBRARIES(prune modNiftiImageIO 
    ITKAlgorithms ITKBasicFilters gsl gslcblas
    ITKCommon ITKIO )
TARGET_LINK_LIBRARIES(concatinate modNiftiImageIO 
    ITKAlgorithms ITKBasicFilters gsl gslcblas
    ITKCommon ITKIO )
TARGET_LINK_LIBRARIES(fmri_timeseries modNiftiImageIO segment 
    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO itkvnl 
    itkvcl vullib)
TARGET_LINK_LIBRARIES(segment gsl gslcblas)
TARGET_LINK_LIBRARIES(unbias ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(printtags modNiftiImageIO ITKAlgorithms 
    ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(copyorigin modNiftiImageIO ITKAlgorithms 
    ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(brainid tools gfortran modNiftiImageIO blas lapack mpi 
    boost_serialization boost_mpi BoldModel ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO mpi gsl gslcblas dysii itkvnl itkvcl vullib)
TARGET_LINK_LIBRARIES(calc_parammap tools gfortran modNiftiImageIO blas lapack mpi 
    boost_serialization boost_mpi BoldModel ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO mpi gsl gslcblas dysii itkvnl itkvcl vullib segment)
TARGET_LINK_LIBRARIES(boldgen tools gfortran blas lapack modNiftiImageIO boost_mpi 
    BoldModel ITKAlgorithms ITKBasicFilters 
    ITKCommon ITKIO gsl gslcblas mpi dysii itkvnl itkvcl vullib)
TARGET_LINK_LIBRARIES(fft tools gfortran blas lapack modNiftiImageIO 
    ITKAlgorithms ITKBasicFilters 
    ITKCommon ITKIO gsl gslcblas itkvnl itkvcl vullib)
TARGET_LINK_LIBRARIES(makesine modNiftiImageIO 
    ITKAlgorithms ITKBasicFilters 
    ITKCommon ITKIO itkvnl itkvcl vullib)

TARGET_LINK_LIBRARIES(readdicom modNiftiImageIO segment 
    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO itkvnl 
    itkvcl vullib)
TARGET_LINK_LIBRARIES(makespline modNiftiImageIO segment 
    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO itkvnl 
    itkvcl vullib)
TARGET_LINK_LIBRARIES(applymask modNiftiImageIO segment 
    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO itkvnl 
    itkvcl vullib)
TARGET_LINK_LIBRARIES(actmap modNiftiImageIO segment tools
    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO itkvnl 
    itkvcl vullib)
TARGET_LINK_LIBRARIES(make_param_image modNiftiImageIO segment tools
    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO itkvnl 
    itkvcl vullib)

TARGET_LINK_LIBRARIES(mean_filter modNiftiImageIO segment 
    ITKAlgorithms ITKBasicFilters ITKCommon ITKIO itkvnl 
    itkvcl vullib)
