#   This file will permit you to add dysii to your project using
#   FIND_PACKAGE(dysii REQUIRED)
#   INCLUDE(${dysii_include} ${dysii_lib})
#

if(NOT dysii_USE_FILE_INCLUDED)
    set(dysii_USE_FILE_INCLUDED 1)

    # Add include directories needed to use dysii.
    include_directories(${dysii_INCLUDE_DIRS})

    # Add link directories needed to use KWWidgets.
    link_directories(${dysii_LIBRARY_DIRS})

endif(NOT dysii_USE_FILE_INCLUDED)

