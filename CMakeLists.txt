cmake_minimum_required(VERSION 3.20)
project(FacetGlass VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FG_BUILD_TESTS "Build core equivalence tests" ON)
option(FG_BUILD_PLUGIN "Build the OpenFX plugin (src/plugin)" ON)

set(OFX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openfx)

# ---------------------------------------------------------------- tests
if(FG_BUILD_TESTS)
  # Reference: the Fuse v1 kernel compiled for the CPU (ground truth renders).
  add_executable(fg_fuse_harness reference/kernel-harness/harness.c)
  if(NOT MSVC)
    target_link_libraries(fg_fuse_harness m)
  endif()
  # Portable core (src/core) on the CPU backend; must match the reference.
  add_executable(fg_core_equiv tests/core_equiv.cpp)
endif()

# ---------------------------------------------------------------- plugin
# TODO (next step, see CLAUDE.md): add src/plugin/*.cpp + the OFX Support
# library as a MODULE library named FacetGlass, output suffix ".ofx",
# placed in FacetGlass.ofx.bundle/Contents/<Win64|MacOS|Linux-x86-64>/.
if(FG_BUILD_PLUGIN AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/FacetGlassPlugin.cpp)
  file(GLOB OFX_SUPPORT_SRC ${OFX_ROOT}/Support/Library/*.cpp)
  file(GLOB FG_PLUGIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*.cpp)
  add_library(FacetGlass MODULE ${FG_PLUGIN_SRC} ${OFX_SUPPORT_SRC})
  target_include_directories(FacetGlass PRIVATE
    ${OFX_ROOT}/include ${OFX_ROOT}/Support/include ${OFX_ROOT}/Support/Library src/core)
  set_target_properties(FacetGlass PROPERTIES PREFIX "" SUFFIX ".ofx")
  if(WIN32)
    set(FG_ARCH Win64)
  elseif(APPLE)
    set(FG_ARCH MacOS)
  else()
    set(FG_ARCH Linux-x86-64)
  endif()
  # The $<1:...> generator expression stops multi-config generators (Visual Studio)
  # from appending a Release/ or Debug/ subfolder, which hosts would not find.
  set_target_properties(FacetGlass PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY $<1:${CMAKE_BINARY_DIR}/FacetGlass.ofx.bundle/Contents/${FG_ARCH}>)
  if(MSVC)
    target_compile_definitions(FacetGlass PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(FacetGlass PRIVATE /W3 /fp:precise)
    # Static CRT: the plugin must load in any host without a VC++ redistributable.
    set_target_properties(FacetGlass PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
endif()
