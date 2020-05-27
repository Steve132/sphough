cmake_minimum_required(VERSION 3.10)


add_library(hough2d
	sparse_hough2d.cpp
	sparse_hough2d.hpp
	naive_hough2d.cpp
	naive_hough2d.hpp
	base_hough2d.cpp
	base_hough2d.hpp
)
set_target_properties(hough2d PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)
target_include_directories(hough2d PUBLIC deps)

if(NOT ANDROID)
add_executable(hough2d_test
	hough2d_test.cpp
)


set_target_properties(hough2d_test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

target_link_libraries(hough2d_test PUBLIC hough2d jpeg)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hough2d_test PUBLIC OpenMP::OpenMP_CXX)
endif()

endif()
