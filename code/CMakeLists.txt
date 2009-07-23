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
    ${GSL_LIBRARY_DIRS})

ADD_LIBRARY(modNiftiImageIO modNiftiImageIO.cxx)
ADD_LIBRARY(segment segment.cpp)
ADD_LIBRARY(BoldModel BoldModel.cpp)

#TARGET_LINK_LIBRARIES(bold particle ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)

ADD_EXECUTABLE(unbias unbias.cpp)
ADD_EXECUTABLE(brainid brainid.cpp)
ADD_EXECUTABLE(printtags printtags.cpp)
ADD_EXECUTABLE(copyorigin copy.cpp)
ADD_EXECUTABLE(fmri_timeseries fmri_timeseries.cpp)
ADD_EXECUTABLE(boldgen boldgen.cpp)
ADD_EXECUTABLE(concatinate concat.cpp)
ADD_EXECUTABLE(prune prune.cpp)

TARGET_LINK_LIBRARIES(prune modNiftiImageIO 
    ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO )
TARGET_LINK_LIBRARIES(concatinate modNiftiImageIO 
    ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO )
TARGET_LINK_LIBRARIES(fmri_timeseries modNiftiImageIO segment 
    ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO itkvnl itkvcl vullib)
TARGET_LINK_LIBRARIES(unbias ITKAlgorithms ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(printtags modNiftiImageIO ITKAlgorithms 
    ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(copyorigin modNiftiImageIO ITKAlgorithms 
    ITKBasicFilters ITKCommon ITKIO)
TARGET_LINK_LIBRARIES(brainid gfortran blas modNiftiImageIO lapack mpi 
    boost_serialization boost_mpi BoldModel ITKAlgorithms ITKBasicFilters
    ITKCommon ITKIO mpi gslcblas gsl dysii itkvnl itkvcl vullib)
TARGET_LINK_LIBRARIES(boldgen gfortran blas lapack modNiftiImageIO boost_mpi 
    BoldModel ITKAlgorithms ITKBasicFilters 
    ITKCommon ITKIO gslcblas gsl mpi dysii itkvnl itkvcl vullib)
