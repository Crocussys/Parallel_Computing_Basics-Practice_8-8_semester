cmake_minimum_required(VERSION 3.14)

project(Parallel_Computing_Basics-Practice_8-8_semester LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)
find_package(MPI REQUIRED)

if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)

    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Parallel_Computing_Basics-Practice_8-8_semester
  main.cpp
)
target_link_libraries(Parallel_Computing_Basics-Practice_8-8_semester Qt${QT_VERSION_MAJOR}::Core MPI::MPI_CXX ${OpenCV_LIBS})

install(TARGETS Parallel_Computing_Basics-Practice_8-8_semester
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
