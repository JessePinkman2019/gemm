#include "gemm.h"
#include "correct_gemm.h"

void correct_gemm(int m, int n, int k, int8_t *a, int lda,
                  int8_t *b, int ldb, int32_t *c, int ldc)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int p = 0; p < k; ++p)
            {
                c[i * ldc + j] += a[i * lda + p] * b[p * ldb + j];
            }
        }
    }
}
