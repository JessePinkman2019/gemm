#include "../source/utils/public.h"
#include "../source/utils/print_matrix.h"
#include "../source/utils/random_matrix.h"
#include "../source/utils/dclock.h"
#include "../source/utils/correct_gemm.h"
#include "../source/utils/compare_matrices.h"
#include "../source/backend/arm/kernel/gemm.h"

int main()
{
    int ret = 0;
    int m(0);
    int n(0);
    int k(0);
    int lda(0);
    int ldb(0);
    int ldc(0);
    double dtime(0.0);
    double gflops(0.0);
    double diff(0.0);
    printf("MY_MMult = [\n");
    for (int matrix_size = MIN_MATRIX_SIZE; matrix_size <= MAX_MATRIX_SIZE; matrix_size += MATRIX_SIZE_STEP)
    {
        m = matrix_size;
        n = matrix_size;
        k = matrix_size;
        gflops = 2.0 * m * n * k * 1.0e-09;
        int8_t *matrix_a = new int8_t[m * k + 8];
        std::fill(matrix_a, matrix_a + m * k + 8, 0);
        int8_t *matrix_b = new int8_t[k * n + 8];
        std::fill(matrix_b, matrix_b + k * n + 8, 0);
        int32_t *matrix_c = new int32_t[m * n + 8];
        std::fill(matrix_c, matrix_c + m * n + 8, 0);
        int32_t *correct_matrix = new int32_t[m * n]{};

        double dtime_best = std::numeric_limits<double>::max();

        for (int rep = 0; rep < NREPEATS; ++rep)
        {
            lda = k;
            ldb = n;
            ldc = n;
            random_matrix(m, k, matrix_a, lda);
            random_matrix(k, n, matrix_b, ldb);
            correct_gemm(m, n, k, matrix_a, lda, matrix_b, ldb, correct_matrix, ldc);

            dtime = dclock();
            gemm_block_packAB_simd(m, n, k, matrix_a, lda, matrix_b, ldb, matrix_c, ldc);
            dtime = dclock() - dtime;

            if (0 == rep)
            {
                dtime_best = dtime;
            }
            else
            {
                dtime_best = dtime < dtime_best ? dtime : dtime_best;
            }
            diff = compare_matrices(m, n, correct_matrix, ldc, matrix_c, ldc);
        }
        delete[] correct_matrix;
        delete[] matrix_c;
        delete[] matrix_b;
        delete[] matrix_a;
        assert(dtime_best > 0 && "dtime_best should be greater than 0");
        // printf("dtime_best: %le\n", dtime_best);

        printf("%d %le %le \n", matrix_size, gflops / dtime_best, diff);
    }
    printf("];\n");
    return ret;
}