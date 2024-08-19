#pragma once

#include "public.h"

void correct_gemm(int m, int n, int k, int8_t *a, int lda,
                  int8_t *b, int ldb, int32_t *c, int ldc);