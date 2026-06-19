cmake_minimum_required(VERSION 3.20)
project(libsimplennet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SIMPLENET_BUILD_TESTS "Build libsimplennet C++ tests" ${PROJECT_IS_TOP_LEVEL})

add_library(simplennet_core STATIC
    src/cpp/simple_predictor.cpp
)
target_include_directories(simplennet_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(simplennet_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(simplennet_c STATIC
    src/cpp/c_api.cpp
)
target_link_libraries(simplennet_c PUBLIC simplennet_core)
target_include_directories(simplennet_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(simplennet_c PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_native bindings/python/native_module.cpp)
    target_link_libraries(_native PRIVATE simplennet_core)
    install(TARGETS _native DESTINATION simplennet)
endif()

find_package(JNI REQUIRED)
add_library(simplennet_native SHARED
    src/cpp/simple_predictor.cpp
    src/cpp/c_api.cpp
    bindings/java/jni_bridge.cpp
)
target_include_directories(simplennet_native PRIVATE ${JNI_INCLUDE_DIRS})
target_include_directories(simplennet_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(simplennet_native PROPERTIES OUTPUT_NAME simplennet_native POSITION_INDEPENDENT_CODE ON)
target_link_libraries(simplennet_native PRIVATE simplennet_core ${JNI_LIBRARIES})
install(TARGETS simplennet_native
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(TARGETS simplennet_core simplennet_c
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES src/libSimpleNNet.py DESTINATION .)

if(SIMPLENET_BUILD_TESTS)
    enable_testing()
    add_executable(simplennet_core_tests tests/cpp/test_core.cpp)
    target_link_libraries(simplennet_core_tests PRIVATE simplennet_core)
    add_test(NAME simplennet_core_tests COMMAND simplennet_core_tests)
endif()
