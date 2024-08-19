#include "compare_matrices.h"

int compare_matrices(int m, int n, const int32_t *a, int lda,
                     const int32_t *b, int ldb)
{
    int ret = 0;
    // 添加断言来验证输入参数的有效性
    assert(m > 0 && "m should be greater than 0");
    assert(n > 0 && "n should be greater than 0");
    assert(a != nullptr && "a should not be null");
    assert(b != nullptr && "b should not be null");
    assert(lda >= n && "lda should be at least n");
    assert(ldb >= n && "ldb should be at least n");

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            assert(a[i * lda + j] == b[i * ldb + j] && "Matrix mismatch");
        }
    }

    return ret;
}