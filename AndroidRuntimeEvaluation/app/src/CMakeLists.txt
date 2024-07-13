cmake_minimum_required(VERSION 3.4.1)

add_library(native-lib
        SHARED
        ${CMAKE_SOURCE_DIR}/main/jni/com_main_tflmodelruntimeevaluation_RuntimeEvaluation.cpp
        )

set_property(TARGET native-lib PROPERTY CXX_STANDARD 17)

add_library(tflite
        STATIC
        IMPORTED
        )

set_target_properties(tflite
        PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/nativeLibs/${CMAKE_ANDROID_ARCH_ABI}/libtensorflowLite.so
        )

set_property(TARGET tflite PROPERTY CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/flatbuffers)
include_directories(${CMAKE_SOURCE_DIR}/include/tensorflow)

target_link_libraries(
                       native-lib
                        tflite )