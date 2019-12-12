cmake_minimum_required(VERSION 2.8)
project("bundle_adjustment")

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/ACMake")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen found:       " ${EIGEN3_INCLUDE_DIR})

find_package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIR})

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
message(STATUS "g2o found:         " ${G2O_INCLUDE_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
message(STATUS "ceres found:       " ${CERES_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    ${CHOLMOD_LIBRARIES}
    ${G2O_CORE_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_TYPES_SBA} ${G2O_STUFF_LIBRARY}
    ${CERES_LIBRARIES})
