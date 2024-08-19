#pragma once

#include <arm_neon.h>
#include "public.h"

void print_int8_matrix(int m, int n, int8_t *a, int lda);
void print_int32_matrix(int m, int n, int32_t *a, int lda);
void packN_16(int k, int n, int8_t *from, int ldb, int8_t *to);
void packZ_16(int m, int k, int8_t *from, int lda, int8_t *to);
void kernel_4x16(int m, int n, int k,
                 int8_t *aligned16_a, int8_t *aligned16_b, int32_t *sc, int ldc);
void gemm_block_packAB_simd(int m, int n, int k, int8_t *matrix_a, int lda,
                            int8_t *matrix_b, int ldb, int32_t *matrix_c, int ldc);
