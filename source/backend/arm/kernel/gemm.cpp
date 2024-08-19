#include "gemm.h"

int8_t *fastMalloc(int size)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 16, size * sizeof(int8_t)) != 0)
    {
        assert(false && "Memory allocation failed!");
        return nullptr;
    }
    return static_cast<int8_t *>(ptr);
}

/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void gemm_block_packAB_simd(int m, int n, int k, int8_t *matrix_a, int lda,
                            int8_t *matrix_b, int ldb, int32_t *matrix_c, int ldc)
{
    assert(m > 0 && "m should be greater than 0");
    assert(n > 0 && "n should be greater than 0");
    assert(k > 0 && "k should be greater than 0");
    assert(matrix_a != nullptr && "matrix_a can not be nullptr");
    assert(matrix_b != nullptr && "matrix_b can not be nullptr");
    assert(matrix_c != nullptr && "matrix_c can not be nullptr");
    assert(lda >= k && "lda must be equal or greater than k");
    assert(ldb >= n && "ldb must be equal or greater than n");
    assert(ldc >= n && "ldc must be equal or greater than n");
    int8_t *aligned16_a = fastMalloc(m * k);
    int8_t *aligned16_b = fastMalloc(k * n);

    int ms, mms, ns, ks;
    int min_m, min_mm, min_n, min_k;
    int l1stride = 1;
    for (ms = 0; ms < m; ms += GEMM_M)
    {
        min_m = m - ms;
        if (min_m > GEMM_M)
        {
            min_m = GEMM_M;
        }

        for (ks = 0; ks < k; ks += min_k)
        {
            min_k = k - ks;
            if (min_k >= (GEMM_K << 1))
            {
                min_k = GEMM_K;
            }
            else if (min_k > GEMM_K)
            {
                min_k = (min_k / 2 + GEMM_UNROLL_K - 1) & ~(GEMM_UNROLL_K - 1);
            }

            // first packB
            min_n = n;
            if (n >= GEMM_N * 2)
            {
                min_n = GEMM_N;
            }
            else if (n > GEMM_N)
            {
                min_n = (min_n / 2 + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1);
            }
            else
            {
                l1stride = 1;
            }

            packN_16(min_k, min_n, matrix_b + ks * ldb, ldb, aligned16_b);

            // micro kernel, split A Block to smaller Panel
            for (mms = ms; mms < ms + min_m; mms += min_mm)
            {
                min_mm = (ms + min_m) - mms;
                if (min_mm >= 3 * GEMM_UNROLL_M)
                {
                    min_mm = 3 * GEMM_UNROLL_M;
                }
                else if (min_mm >= 2 * GEMM_UNROLL_M)
                {
                    min_mm = 2 * GEMM_UNROLL_M;
                }
                else if (min_mm > GEMM_UNROLL_M)
                {
                    min_mm = GEMM_UNROLL_M;
                }

                // coninueous packA
                packZ_16(min_mm, min_k, matrix_a + mms * lda + ks, lda, aligned16_a + min_k * (mms - ms) * l1stride);

                kernel_4x16(min_mm, min_n, min_k, aligned16_a + l1stride * min_k * (mms - ms), aligned16_b, matrix_c + mms * ldc, ldc);
            }

            // the first B Block has been packed, proc the others
            for (ns = min_n; ns < n; ns += min_n)
            {
                min_n = n - ns;
                if (min_n >= GEMM_N * 2)
                {
                    min_n = GEMM_N;
                }
                else if (min_n > GEMM_N)
                {
                    min_n = (min_n / 2 + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1);
                }

                packN_16(min_k, min_n, matrix_b + ns + ldb * ks, ldb, aligned16_b);

                kernel_4x16(min_m, min_n, min_k, aligned16_a, aligned16_b, matrix_c + ms * ldc + ns, ldc);
            }
        }
    }

    free(aligned16_a);
    free(aligned16_b);
}

void kernel_sub_v1(int m, int n, int k, int8_t *aligned16_a, int8_t *aligned16_b, int32_t *sc, int ldc)
{
    assert(aligned16_a != nullptr && "aligned16_a should not be null");
    assert(aligned16_b != nullptr && "aligned16_b should not be null");
    assert(sc != nullptr && "sc should not be null");
    assert(ldc >= n && "ldc should be at least n");
    int8_t *a = aligned16_a;
    int32_t *c = sc;

    for (int i = 0; i < m; ++i)
    {
        int8_t *b = aligned16_b;

        for (int j = 0; j < n; ++j)
        {
            int32x4_t c_vec = vdupq_n_s32(0); // 初始化累加向量

            for (int x = 0; x < k; x += 16)
            {
                // 加载 a 和 b 的 16 个元素到 NEON 寄存器
                int8x16_t a_vec = vld1q_s8(a + x);
                int8x16_t b_vec = vld1q_s8(b + x);

                // 转换为 16 位整数
                int16x8_t a_vec_low = vmovl_s8(vget_low_s8(a_vec));
                int16x8_t a_vec_high = vmovl_s8(vget_high_s8(a_vec));
                int16x8_t b_vec_low = vmovl_s8(vget_low_s8(b_vec));
                int16x8_t b_vec_high = vmovl_s8(vget_high_s8(b_vec));

                // 乘法并累加
                c_vec = vmlal_s16(c_vec, vget_low_s16(a_vec_low), vget_low_s16(b_vec_low));
                c_vec = vmlal_s16(c_vec, vget_high_s16(a_vec_low), vget_high_s16(b_vec_low));
                c_vec = vmlal_s16(c_vec, vget_low_s16(a_vec_high), vget_low_s16(b_vec_high));
                c_vec = vmlal_s16(c_vec, vget_high_s16(a_vec_high), vget_high_s16(b_vec_high));
            }

            // 将累加结果存储到 c
            c[j] += vaddvq_s32(c_vec);

            b += k;
        }
        a += k;
        c += ldc;
    }
}

// get c[m, n] output
void kernel_mn(int m, int n, int k, int8_t *aligned16_a, int8_t *aligned16_b, int32_t *sc, int ldc)
{
    // sum_all( A4xsubk * Bsubkx4 )
    int8_t *a = aligned16_a, *b = aligned16_b;
    int shift = 4;
    while (k > 0)
    {
        int repeat = k >> shift;
        int step = 1 << shift;
        for (int i = 0; i < repeat; ++i)
        {
            kernel_sub_v1(m, n, step, a, b, sc, ldc);
            a += m * step;
            b += n * step;
        }
        k -= (repeat << shift);
        shift--;
    }
}

// proc m lines
void kernel_m(int m, int n, int k, int8_t *aligned16_a, int8_t *aligned16_b, int32_t *sc, int ldc)
{
    // m == 4
    int nn = n;
    int8_t *b = aligned16_b;
    int32_t *c = sc;

    while (nn >= 4)
    {
        kernel_mn(m, 4, k, aligned16_a, b, c, ldc);
        b += 4 * k;
        c += 4;
        nn -= 4;
    };

    while (nn >= 2)
    {
        kernel_mn(m, 2, k, aligned16_a, b, c, ldc);
        b += 2 * k;
        c += 2;
        nn -= 2;
    };

    while (nn >= 1)
    {
        kernel_mn(m, 1, k, aligned16_a, b, c, ldc);
        b += 1 * k;
        c += 1;
        nn -= 1;
    }
}

void kernel_4x16(int m, int n, int k,
                 int8_t *aligned16_a, int8_t *aligned16_b, int32_t *sc, int ldc)
{
    int mm = m;
    int8_t *a = aligned16_a;
    int32_t *c = sc;
    while (mm >= 4)
    {
        kernel_m(4, n, k, a, aligned16_b, c, ldc);
        a += 4 * k;
        c += 4 * ldc;
        mm -= 4;
    };

    while (mm >= 2)
    {
        kernel_m(2, n, k, a, aligned16_b, c, ldc);
        a += 2 * k;
        c += 2 * ldc;
        mm -= 2;
    };

    while (mm >= 1)
    {
        kernel_m(1, n, k, a, aligned16_b, c, ldc);
        a += k;
        c += ldc;
        mm--;
    };
}

void packZ_sub(int8_t *from, int lda, int8_t *to, int m, int n, int repeat)
{
    int8_t *ptr;
    for (int rep = 0; rep < repeat; ++rep)
    {
        ptr = from;
        for (int i = 0; i < m; ++i)
        {
            memcpy(to, ptr, n * sizeof(int8_t));
            to += n;
            ptr += lda;
        }
        from += n;
    }
}

// pack4x16
void packZ_16(int m, int k, int8_t *from, int lda, int8_t *to)
{

    // TODO to be optimize
    int col, proc_col;
    int8_t *a_offset = from;
    int8_t *col_offset = a_offset;
    int8_t *row_offset = col_offset;
    int8_t *c = to;

    int shift = 2;
    while (m > 0)
    {
        assert(shift >= 0);
        int num_of_rowblocks = m >> shift;
        const int proc_row = 1 << shift;
        while (num_of_rowblocks > 0)
        {
            col = k;
            col_offset = row_offset;

            // proc 16x col
            int repeat = col >> 4;
            proc_col = repeat << 4;

            packZ_sub(col_offset, lda, c, proc_row, 16, repeat);
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // proc 8x col
            repeat = col >> 3;
            proc_col = repeat << 3;

            packZ_sub(col_offset, lda, c, proc_row, 8, repeat);
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // proc 4x col
            repeat = col >> 2;
            proc_col = repeat << 2;

            packZ_sub(col_offset, lda, c, proc_row, 4, repeat);
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // proc 2x col
            repeat = col >> 1;
            proc_col = repeat << 1;

            packZ_sub(col_offset, lda, c, proc_row, 2, repeat);
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // prco 1x col
            packZ_sub(col_offset, lda, c, proc_row, 1, col);
            col_offset += col;
            c += proc_row * col;

            row_offset += proc_row * lda;
            --num_of_rowblocks;
        };
        a_offset += ((m >> shift) << shift) * lda;
        row_offset = a_offset;
        m -= ((m >> shift) << shift);
        --shift;
    }
}

void packN_sub(int8_t *from, int ldb, int8_t *to, int m, int n, int repeat)
{
    int8_t *ctemp[16] = {0};

    for (int r = 0; r < repeat; ++r)
    {

        if (m == 1)
        {
            ctemp[0] = from;
        }
        else if (m == 2)
        {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
        }
        else if (m == 4)
        {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
            ctemp[2] = from + 2 * ldb;
            ctemp[3] = from + 3 * ldb;
        }
        else if (m == 8)
        {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
            ctemp[2] = from + 2 * ldb;
            ctemp[3] = from + 3 * ldb;
            ctemp[4] = from + 4 * ldb;
            ctemp[5] = from + 5 * ldb;
            ctemp[6] = from + 6 * ldb;
            ctemp[7] = from + 7 * ldb;
        }
        else if (m == 16)
        {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
            ctemp[2] = from + 2 * ldb;
            ctemp[3] = from + 3 * ldb;
            ctemp[4] = from + 4 * ldb;
            ctemp[5] = from + 5 * ldb;
            ctemp[6] = from + 6 * ldb;
            ctemp[7] = from + 7 * ldb;
            ctemp[8] = from + 8 * ldb;
            ctemp[9] = from + 9 * ldb;
            ctemp[10] = from + 10 * ldb;
            ctemp[11] = from + 11 * ldb;
            ctemp[12] = from + 12 * ldb;
            ctemp[13] = from + 13 * ldb;
            ctemp[14] = from + 14 * ldb;
            ctemp[15] = from + 15 * ldb;
        }
        else
        {
            assert(0);
        }

        for (int i = 0; i < n; ++i)
        {
            if (m == 1)
            {
                to[0] = ctemp[0][i];
            }
            else if (m == 2)
            {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
            }
            else if (m == 4)
            {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
                to[2] = ctemp[2][i];
                to[3] = ctemp[3][i];
            }
            else if (m == 8)
            {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
                to[2] = ctemp[2][i];
                to[3] = ctemp[3][i];
                to[4] = ctemp[4][i];
                to[5] = ctemp[5][i];
                to[6] = ctemp[6][i];
                to[7] = ctemp[7][i];
            }
            else if (m == 16)
            {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
                to[2] = ctemp[2][i];
                to[3] = ctemp[3][i];
                to[4] = ctemp[4][i];
                to[5] = ctemp[5][i];
                to[6] = ctemp[6][i];
                to[7] = ctemp[7][i];
                to[8] = ctemp[8][i];
                to[9] = ctemp[9][i];
                to[10] = ctemp[10][i];
                to[11] = ctemp[11][i];
                to[12] = ctemp[12][i];
                to[13] = ctemp[13][i];
                to[14] = ctemp[14][i];
                to[15] = ctemp[15][i];
            }
            else
            {
                assert(0);
            }
            to += m;
        }
        from += ldb * m;
    }
}

// pack16x4
void packN_16(int k, int n, int8_t *from, int ldb, int8_t *to)
{
    int row;
    int proc_row;

    int8_t *a_offset = from;
    int8_t *a_offset1 = a_offset;
    int8_t *c = to;

    int shift = 2;
    while (n > 0)
    {
        assert(shift >= 0);
        int num_of_colblock = n >> shift;
        const int proc_col = 1 << shift;
        while (num_of_colblock > 0)
        {
            row = k;

            // proc 16x row
            int repeat = row >> 4;
            proc_row = repeat << 4;
            packN_sub(a_offset1, ldb, c, 16, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 8x row
            repeat = row >> 3;
            proc_row = repeat << 3;
            packN_sub(a_offset1, ldb, c, 8, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 4x row
            repeat = row >> 2;
            proc_row = repeat << 2;
            packN_sub(a_offset1, ldb, c, 4, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 2x row
            repeat = row >> 1;
            proc_row = repeat << 1;
            packN_sub(a_offset1, ldb, c, 2, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 1x row
            repeat = row;
            proc_row = repeat;
            packN_sub(a_offset1, ldb, c, 1, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_col * row;
            row -= proc_row;

            --num_of_colblock;
            a_offset += proc_col;
            a_offset1 = a_offset;
        };
        n -= ((n >> shift) << shift);
        --shift;
    }
}