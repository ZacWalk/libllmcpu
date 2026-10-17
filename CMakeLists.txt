# nnc — Neural Net Compiler
#
# Targets:
#   libnnc       static library (nnc.lib / libnnc.a): GGUF loading, the JIT,
#                the kernels, the forward pass, the tokenizer and sessions.
#                Its public API is the one header src/libnnc/nnc.h.
#   nnc          the command-line app (src/appnnc): chat REPL, model picker
#                and downloader, inspection tools, and the self-tests (--test).
#   nnc-minimal, nnc-embed, nnc-function-call
#                optional plain-C consumers of nnc.h (examples/): one chat
#                message, semantic search, and a FunctionGemma call.
#
#   cmake --preset release && cmake --build --preset release   -> exe/nnc[.exe]
#   cmake --preset debug   && cmake --build --preset debug     -> exe/nnc-d[.exe]
#
# Using libnnc from another CMake project:
#   add_subdirectory(path/to/nnc)    # the app and examples are skipped there
#   target_link_libraries(myapp PRIVATE nnc::nnc)
#
# Requires: CMake 3.21+, Ninja, and MSVC v143+ / g++ 10+ / clang++ 12+.
# Target CPU baseline is x86-64 with AVX2 + FMA.

cmake_minimum_required(VERSION 3.21)

project(nnc VERSION 0.3.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NNC_BUILD_APP "Build the nnc command-line app" ${PROJECT_IS_TOP_LEVEL})
option(NNC_BUILD_EXAMPLES "Build the examples" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# ---- compiler settings shared by every target ------------------------------

if(MSVC)
	set(NNC_COMMON_DEFINITIONS _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE $<$<CONFIG:Debug>:_DEBUG>)
	# No /GL (whole-program optimisation): its objects link only with the exact
	# toolset that built them, which would make an installed nnc.lib unusable
	# elsewhere, and cross-file inlining could carry AVX2 code into
	# baseline.cpp's CPU gate.
	set(NNC_COMMON_OPTIONS
		/W3
		/permissive-
		/Zc:__cplusplus
		/utf-8 # sources contain UTF-8 literals (U+2581, µ)
		/Zi
		$<$<NOT:$<CONFIG:Debug>>:/O2;/Oi;/Ot;/Gy>
	)
	set(NNC_LINK_OPTIONS /DEBUG $<$<NOT:$<CONFIG:Debug>>:/OPT:REF;/OPT:ICF>)
	set(NNC_SIMD_FLAGS /arch:AVX2)
	set(NNC_FAST_MATH /fp:fast)
else()
	set(NNC_COMMON_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>)
	set(NNC_COMMON_OPTIONS
		-Wall -Wextra
		-Wno-unused-parameter
		-Wno-missing-field-initializers
		$<$<CONFIG:Debug>:-O0;-g>
		$<$<NOT:$<CONFIG:Debug>>:-O3>
	)
	set(NNC_LINK_OPTIONS)
	set(NNC_SIMD_FLAGS -mavx2 -mfma -mf16c)
	set(NNC_FAST_MATH $<$<NOT:$<CONFIG:Debug>>:-ffast-math;-fno-finite-math-only>)
endif()

# ---- libnnc -----------------------------------------------------------------
#
# Source lists are explicit, not globbed: a new .cpp does nothing until it is
# listed here.

# Code that must run on any x86-64 CPU: the check that decides whether nnc
# can run at all, and every public function a caller can reach before it has
# passed (see baseline.cpp). Built without AVX flags.
set(LIBNNC_PORTABLE_SOURCES
	src/libnnc/baseline.cpp
	src/libnnc/sys_linux.cpp
	src/libnnc/sys_win.cpp
)

# Everything else: kernels, model and runtime, built for AVX2 + FMA.
set(LIBNNC_SIMD_SOURCES
	src/libnnc/emitter_avx2.cpp
	src/libnnc/emitter_x64.cpp
	src/libnnc/gguf.cpp
	src/libnnc/jit_buffer.cpp
	src/libnnc/jit_kernel.cpp
	src/libnnc/jit_ops.cpp
	src/libnnc/model.cpp
	src/libnnc/nn_ops.cpp
	src/libnnc/nnc.cpp
	src/libnnc/runtime.cpp
	src/libnnc/session.cpp
	src/libnnc/tokenizer.cpp
)
set_source_files_properties(${LIBNNC_SIMD_SOURCES} PROPERTIES COMPILE_OPTIONS "${NNC_SIMD_FLAGS}")

add_library(libnnc STATIC ${LIBNNC_PORTABLE_SOURCES} ${LIBNNC_SIMD_SOURCES})
add_library(nnc::nnc ALIAS libnnc)
set_target_properties(libnnc PROPERTIES OUTPUT_NAME nnc EXPORT_NAME nnc) # nnc.lib / libnnc.a
target_include_directories(libnnc PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/libnnc>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(libnnc PUBLIC Threads::Threads)
if(WIN32)
	target_link_libraries(libnnc PRIVATE winhttp shell32)
endif()
target_compile_definitions(libnnc PRIVATE ${NNC_COMMON_DEFINITIONS} NNC_VERSION="${PROJECT_VERSION}")
target_compile_options(libnnc PRIVATE ${NNC_COMMON_OPTIONS} ${NNC_FAST_MATH})

install(TARGETS libnnc EXPORT nncTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES src/libnnc/nnc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
	cmake/nncConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/nncConfig.cmake"
	INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nnc"
)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/nncConfigVersion.cmake"
	VERSION "${PROJECT_VERSION}"
	COMPATIBILITY SameMajorVersion
)
install(EXPORT nncTargets FILE nncTargets.cmake NAMESPACE nnc::
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nnc"
)
install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/nncConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/nncConfigVersion.cmake"
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nnc"
)

# ---- the nnc app ------------------------------------------------------------

if(NNC_BUILD_APP)
	add_executable(nnc
		src/appnnc/chat.cpp
		src/appnnc/cli.cpp
		src/appnnc/devtools.cpp
		src/appnnc/embed.cpp
		src/appnnc/main.cpp
		src/appnnc/models.cpp
		src/appnnc/platform.cpp
		src/appnnc/tests.cpp
		src/appnnc/ui.cpp
	)
	target_link_libraries(nnc PRIVATE libnnc)
	target_compile_definitions(nnc PRIVATE ${NNC_COMMON_DEFINITIONS} $<$<BOOL:${MSVC}>:_CONSOLE>)
	target_compile_options(nnc PRIVATE ${NNC_COMMON_OPTIONS})
	target_link_options(nnc PRIVATE ${NNC_LINK_OPTIONS})
	# Debug builds are nnc-d so both binaries can live in exe/ side by side.
	set_target_properties(nnc PROPERTIES
		OUTPUT_NAME nnc
		OUTPUT_NAME_DEBUG nnc-d
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exe"
	)
	foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
		string(TOUPPER "${cfg}" cfg_upper)
		set_target_properties(nnc PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_SOURCE_DIR}/exe")
	endforeach()

	enable_testing()
	add_test(NAME nnc-selftest COMMAND nnc --test)
	set_tests_properties(nnc-selftest PROPERTIES LABELS nnc)
endif()

# ---- examples ---------------------------------------------------------------

if(NNC_BUILD_EXAMPLES)
	# Plain C against nnc.h; linked with the C++ driver because libnnc is C++.
	foreach(example minimal embed function_call)
		string(REPLACE "_" "-" target "nnc-${example}")
		add_executable(${target} examples/${example}.c)
		target_link_libraries(${target} PRIVATE libnnc)
		target_compile_options(${target} PRIVATE $<IF:$<BOOL:${MSVC}>,/W3,-Wall>)
		target_link_options(${target} PRIVATE ${NNC_LINK_OPTIONS})
		set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
	endforeach()
endif()
