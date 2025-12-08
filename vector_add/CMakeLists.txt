cmake_minimum_required(VERSION 3.20)

# Project: vector_add (CUDA + C++14)
project(vector_add LANGUAGES CXX CUDA)

# Ensure C++14 and CUDA 14-ish compatible flags (C++14 requested)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Visual Studio generator specifics (optional but useful)
# Compile CUDA with RelWithDebInfo by default for VS
if(MSVC)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
endif()

# Sources
set(SOURCES
    kernel.cu
)

# Create executable
add_executable(vector_add ${SOURCES})

# Enable separable compilation for CUDA (recommended for .cu files)
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# On Windows with VS, make sure runtime is static or default as needed
# Uncomment if you want to force static runtime for MSVC:
# set_property(TARGET vector_add PROPERTY
#     MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
# )

# Define any necessary compile options
# Adjust architecture as needed (e.g., 75 for Turing, 86 for Ampere, 89 for Ada)
# You can add multiple gencodes; here we pick a common default.
set(CUDA_ARCH "native")
if(CUDA_ARCH STREQUAL "native")
    # Let nvcc pick a suitable arch; or set one explicitly:
    # target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_86,code=sm_86>)
else()
    target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}>)
endif()

# Treat warnings as needed
# target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)