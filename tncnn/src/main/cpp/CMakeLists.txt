# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5.0)
project(NcnnHarmony)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/include/ncnn)
aux_source_directory(. SRC_LIST)
add_library(
        tncnn
        SHARED
        ${SRC_LIST}
)

target_link_libraries(tncnn PUBLIC ncnn libace_napi.z.so libhilog_ndk.z.so librawfile.z.so)
