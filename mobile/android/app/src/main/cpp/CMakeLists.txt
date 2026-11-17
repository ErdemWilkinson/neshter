cmake_minimum_required(VERSION 3.22)
project(neshter_llama LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp source tree (pinned release; see mobile/README.md). Override with -DLLAMA_SRC=...
set(LLAMA_SRC "D:/neshter/llama.cpp-src" CACHE PATH "llama.cpp source tree")

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

if(ANDROID)
    # One libggml-cpu-*.so per ARM feature level (dotprod, i8mm, SVE...); the fastest one the phone supports is
    # picked when the app starts. KleidiAI adds Arm's tuned matrix kernels on arm64.
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
        set(GGML_OPENMP ON CACHE BOOL "" FORCE)
    else()
        set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
        set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    endif()
endif()

add_subdirectory(${LLAMA_SRC} llama-build)

add_library(neshter_engine STATIC neshter_llama.cpp)
target_include_directories(neshter_engine PUBLIC ${CMAKE_CURRENT_LIST_DIR})
# llama-model.h (internal): the lens reads the model's final norm and output matrix.
target_include_directories(neshter_engine PRIVATE ${LLAMA_SRC}/src)
target_link_libraries(neshter_engine PUBLIC llama ggml)

if(ANDROID)
    add_library(neshter_llama SHARED jni_bridge.cpp)
    target_link_libraries(neshter_llama PRIVATE neshter_engine android log)
else()
    add_executable(nl_check nl_check.cpp)
    target_link_libraries(nl_check PRIVATE neshter_engine)
endif()
