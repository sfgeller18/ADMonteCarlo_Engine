# Try to find the GMP library
# See https://gmplib.org/
#
# This module will find the GMP library without requiring a minimum version.
#
# Once done this will define
#
#  GMP_FOUND - system has GMP lib
#  GMP_INCLUDES - the GMP include directory
#  GMP_LIBRARIES - the GMP library
#  GMP_VERSION - GMP version (if available)
# Copyright (c) 2023 Your Name <your.email@example.com>
# Redistribution and use is allowed according to the terms of the BSD license.
# Set GMP_INCLUDES
find_path(GMP_INCLUDES
  NAMES
  gmp.h
  PATHS
  /usr/include
  /usr/local/include
  $ENV{GMPDIR}
  ${INCLUDE_INSTALL_DIR}
)
if(GMP_INCLUDES)
  # Set GMP_VERSION if available
  
  file(READ "${GMP_INCLUDES}/gmp.h" _gmp_version_header)
  
  string(REGEX MATCH "define[ \t]+__GNU_MP_VERSION[ \t]+\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\"" _gmp_version_match "${_gmp_version_header}")
  set(GMP_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(GMP_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(GMP_VERSION_PATCH "${CMAKE_MATCH_3}")
  
  set(GMP_VERSION ${GMP_VERSION_MAJOR}.${GMP_VERSION_MINOR}.${GMP_VERSION_PATCH})
endif()
# Set GMP_LIBRARIES
find_library(GMP_LIBRARIES gmp PATHS /usr/lib /usr/local/lib $ENV{GMPDIR} ${LIB_INSTALL_DIR})
# Epilogue
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG
                                  GMP_INCLUDES GMP_LIBRARIES GMP_VERSION)
mark_as_advanced(GMP_INCLUDES GMP_LIBRARIES)
