cmake_minimum_required(VERSION 3.20)
project(rgb2gray LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find OpenCV (required by rgb2gray)
find_package(OpenCV REQUIRED)

# Sources
add_executable(rgb2gray
    kernel.cu
)

# CUDA properties
set_target_properties(rgb2gray PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include and link OpenCV
target_include_directories(rgb2gray PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rgb2gray PRIVATE ${OpenCV_LIBS})

# Optional: set GPU architecture (adjust as needed)
# target_compile_options(rgb2gray PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_86,code=sm_86>)