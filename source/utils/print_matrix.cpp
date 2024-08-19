#include "print_matrix.h"

void print_int8_matrix(int m, int n, const int8_t *a, int lda)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%d\t", a[i * lda + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_int32_matrix(int m, int n, const int32_t *a, int lda)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%d\t", a[i * lda + j]);
        }
        printf("\n");
    }
    printf("\n");
}
