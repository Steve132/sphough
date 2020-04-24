cmake_minimum_required(VERSION 3.10)


add_library(sparse_hough2d
	sparse_hough2d.cpp
	sparse_hough2d.hpp
)
set_target_properties(sparse_hough2d PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
	COMPILE_FLAGS "-march=native -Wignored-attributes"
)

add_executable(sphough2d_test
	sphough2d_test.cpp
)

set_target_properties(sphough2d_test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    COMPILE_FLAGS "-march=native -Wignored-attributes"
)

target_include_directories(sphough2d_test PUBLIC deps)
target_link_libraries(sphough2d_test PUBLIC sparse_hough2d X11 jpeg)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sphough2d_test PUBLIC OpenMP::OpenMP_CXX)
endif()
