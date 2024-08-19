# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "CXX"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_CXX
  "/home/lixiang/code/gemm/source/backend/arm/kernel/gemm.cpp" "/home/lixiang/code/gemm/test/build/CMakeFiles/main.x.dir/home/lixiang/code/gemm/source/backend/arm/kernel/gemm.cpp.o"
  "/home/lixiang/code/gemm/source/utils/compare_matrices.cpp" "/home/lixiang/code/gemm/test/build/CMakeFiles/main.x.dir/home/lixiang/code/gemm/source/utils/compare_matrices.cpp.o"
  "/home/lixiang/code/gemm/source/utils/correct_gemm.cpp" "/home/lixiang/code/gemm/test/build/CMakeFiles/main.x.dir/home/lixiang/code/gemm/source/utils/correct_gemm.cpp.o"
  "/home/lixiang/code/gemm/source/utils/dclock.cpp" "/home/lixiang/code/gemm/test/build/CMakeFiles/main.x.dir/home/lixiang/code/gemm/source/utils/dclock.cpp.o"
  "/home/lixiang/code/gemm/source/utils/print_matrix.cpp" "/home/lixiang/code/gemm/test/build/CMakeFiles/main.x.dir/home/lixiang/code/gemm/source/utils/print_matrix.cpp.o"
  "/home/lixiang/code/gemm/source/utils/random_matrix.cpp" "/home/lixiang/code/gemm/test/build/CMakeFiles/main.x.dir/home/lixiang/code/gemm/source/utils/random_matrix.cpp.o"
  "/home/lixiang/code/gemm/test/main.cpp" "/home/lixiang/code/gemm/test/build/CMakeFiles/main.x.dir/main.cpp.o"
  )
set(CMAKE_CXX_COMPILER_ID "GNU")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_CXX
  "GEMM_K=128"
  "GEMM_M=128"
  "GEMM_N=128"
  "GEMM_UNROLL_K=16"
  "GEMM_UNROLL_M=4"
  "GEMM_UNROLL_N=4"
  "MATRIX_SIZE_STEP=128"
  "MAX_MATRIX_SIZE=1024"
  "MIN_MATRIX_SIZE=128"
  "NREPEATS=1"
  )

# The include file search paths:
set(CMAKE_CXX_TARGET_INCLUDE_PATH
  "../../source/backend/arm/kernel"
  "../../source/utils"
  )

# Targets to which this target links.
set(CMAKE_TARGET_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
