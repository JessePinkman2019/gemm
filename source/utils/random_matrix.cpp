#include "random_matrix.h"

void random_matrix(int m, int n, int8_t *a, int lda)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int8_t> distribution(-128, 127);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            a[i * lda + j] = distribution(generator);
        }
    }
}
