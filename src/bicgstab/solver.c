#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

/* ---------- 小工具：分母保护 ---------- */
#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

static inline double safe_div(double num, double den, double eps)
{
    if (fabs(den) <= eps)
        return NAN;
    return num / den;
}

/* 计算当前 rank 负责的 [start, end) 段（用于全局长度 N 的向量） */
static inline void my_range(int N, int *start, int *end)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int base = N / size, rem = N % size;
    int local_N = base + (rank < rem ? 1 : 0);
    *start = rank * base + (rank < rem ? rank : rem);
    *end = *start + local_N;
}

/* y_local = A_local * x  （A_local: local_rows x N）*/
void gemv(double *__restrict y_local,
          const double *__restrict A_local,
          const double *__restrict x,
          int N, int local_rows)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < local_rows; i++)
    {
        double sum = 0.0;
        const double *row = A_local + (size_t)i * N;
        for (int j = 0; j < N; j++)
            sum += row[j] * x[j];
        y_local[i] = sum;
    }
}

/* 点积：支持不整除 + OpenMP 本地归约 + 一次 Allreduce */
double dot_product(const double *__restrict x,
                   const double *__restrict y,
                   int N)
{
    int s, e;
    my_range(N, &s, &e);
    double local = 0.0;
#pragma omp parallel for reduction(+ : local) schedule(static)
    for (int i = s; i < e; i++)
        local += x[i] * y[i];

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

/* 一次性做两个点积的 Allreduce： out[0]=(a·b), out[1]=(c·d) */
static inline void dots2(const double *__restrict a, const double *__restrict b,
                         const double *__restrict c, const double *__restrict d,
                         int N, double out[2])
{
    int s, e;
    my_range(N, &s, &e);
    double loc0 = 0.0, loc1 = 0.0;
#pragma omp parallel for reduction(+ : loc0, loc1) schedule(static)
    for (int i = s; i < e; i++)
    {
        loc0 += a[i] * b[i];
        loc1 += c[i] * d[i];
    }
    double buf[2] = {loc0, loc1};
    MPI_Allreduce(buf, out, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/* 预条件子（对角近似的 M^{-1}），带对角保护 */
void precondition(const double *__restrict A,
                  double *__restrict K2_inv, int N)
{
    const double eps_diag = 1e-12;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        double aii = A[(size_t)i * N + i];
        if (fabs(aii) < eps_diag)
            aii = (aii >= 0.0 ? eps_diag : -eps_diag);
        K2_inv[i] = 1.0 / aii;
    }
}

/* z = K2_inv * r */
void precondition_apply(double *__restrict z,
                        const double *__restrict K2_inv,
                        const double *__restrict r, int N)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        z[i] = K2_inv[i] * r[i];
}

/* ---------------------------- BiCGSTAB 主体 ---------------------------- */
int bicgstab(int N, double *A, double *b, double *x, int max_iter, double tol)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* 行分块（支持 N 不整除 size） */
    int base = N / size, rem = N % size;
    int local_rows = base + (rank < rem ? 1 : 0);

    /* 每 rank 的行数与位移（以“行”为单位） */
    int *rows_counts = (int *)malloc(size * sizeof(int));
    int *rows_displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r)
        rows_counts[r] = base + (r < rem ? 1 : 0);
    rows_displs[0] = 0;
    for (int r = 1; r < size; ++r)
        rows_displs[r] = rows_displs[r - 1] + rows_counts[r - 1];

    /* A 的 Scatterv 需要元素个数（行数 * N）与位移（元素单位） */
    int *A_counts = (int *)malloc(size * sizeof(int));
    int *A_displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r)
        A_counts[r] = rows_counts[r] * N;
    A_displs[0] = 0;
    for (int r = 1; r < size; ++r)
        A_displs[r] = A_displs[r - 1] + A_counts[r - 1];

    /* 确保所有 rank 都有 b 与 x（全量副本） */
    int b_alloc_local = 0, x_alloc_local = 0;
    if (rank != 0)
    {
        if (b == NULL)
        {
            b = (double *)malloc((size_t)N * sizeof(double));
            b_alloc_local = 1;
        }
        if (x == NULL)
        {
            x = (double *)calloc((size_t)N, sizeof(double));
            x_alloc_local = 1;
        }
    }
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 本地块与临时向量 */
    double *A_local = (double *)malloc((size_t)local_rows * N * sizeof(double));
    double *v_local = (double *)calloc((size_t)local_rows, sizeof(double));
    double *t_local = (double *)calloc((size_t)local_rows, sizeof(double));

    MPI_Scatterv(A, A_counts, A_displs, MPI_DOUBLE,
                 A_local, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* 全局长度的向量（各 rank 都持有一份） */
    double *r = (double *)calloc((size_t)N, sizeof(double));
    double *r_hat = (double *)calloc((size_t)N, sizeof(double));
    double *p = (double *)calloc((size_t)N, sizeof(double));
    double *v = (double *)calloc((size_t)N, sizeof(double)); /* 仍用于收集 v */
    double *s = (double *)calloc((size_t)N, sizeof(double));
    double *h = (double *)calloc((size_t)N, sizeof(double));
    double *t = (double *)calloc((size_t)N, sizeof(double)); /* 仍用于收集 t */
    double *y = (double *)calloc((size_t)N, sizeof(double));
    double *z = (double *)calloc((size_t)N, sizeof(double));
    double *K2_inv = (double *)calloc((size_t)N, sizeof(double));

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho = 1.0, beta = 1.0;

    const double tol_sq = tol * tol;
    const double eps_den = 1e-30;
    const int k_recomp = 200; /* 每 k 步重算 r */

    /* 预条件子（在 root 构造 K2_inv，然后广播） */
    if (rank == 0)
        precondition(A, K2_inv, N);
    MPI_Bcast(K2_inv, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---------- 初始化 r0 = b - A x0（仅用本地块更新，去掉 Allgatherv） ---------- */
    double *Ax_local = (double *)calloc((size_t)local_rows, sizeof(double));
    gemv(Ax_local, A_local, x, N, local_rows);

    int s_idx, e_idx;
    my_range(N, &s_idx, &e_idx);
    int locN = e_idx - s_idx;
#pragma omp parallel for schedule(static)
    for (int li = 0; li < locN; ++li)
    {
        int i = s_idx + li;
        r[i] = b[i] - Ax_local[li];
    }

    double r0_norm2 = dot_product(r, r, N);
    if (rank == 0 && r0_norm2 == 0.0)
    {
        free(Ax_local);
        free(A_local);
        free(v_local);
        free(t_local);
        free(r);
        free(r_hat);
        free(p);
        free(v);
        free(s);
        free(h);
        free(t);
        free(y);
        free(z);
        free(K2_inv);
        if (b_alloc_local)
            free(b);
        if (x_alloc_local)
            free(x);
        free(rows_counts);
        free(rows_displs);
        free(A_counts);
        free(A_displs);
        return 0;
    }

    memcpy(r_hat, r, (size_t)N * sizeof(double));
    rho = dot_product(r_hat, r, N);
    memcpy(p, r, (size_t)N * sizeof(double));

    int iter;
    for (iter = 1; iter <= max_iter; iter++)
    {
        /* y = M^{-1} p */
        precondition_apply(y, K2_inv, p, N);

        /* v = A y  —— 本地 gemv + Allgatherv 收集为全局 v */
        gemv(v_local, A_local, y, N, local_rows);
        MPI_Allgatherv(v_local, local_rows, MPI_DOUBLE,
                       v, rows_counts, rows_displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        /* alpha 所需的点积 (r_hat, v) */
        double rv = dot_product(r_hat, v, N);
        double alpha_try = safe_div(rho, rv, eps_den);
        if (unlikely(!isfinite(alpha_try)))
        {
            if (rank == 0)
                fprintf(stderr, ">>> Breakdown: (r_hat, v) ~ 0 at iter %d, soft-restart\n", iter);
            memcpy(r_hat, r, (size_t)N * sizeof(double));
            rho = dot_product(r_hat, r, N);
            memcpy(p, r, (size_t)N * sizeof(double));
            continue;
        }
        alpha = alpha_try;

        /* h = x + alpha y */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            h[i] = x[i] + alpha * y[i];

        /* s = r - alpha v */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            s[i] = r[i] - alpha * v[i];

        /* ====== 关键：把 ||s||^2 的 Allreduce 改成 Iallreduce，并与本地计算重叠 ====== */
        double s_loc2 = 0.0;
#pragma omp parallel for reduction(+ : s_loc2) schedule(static)
        for (int i = s_idx; i < e_idx; i++)
            s_loc2 += s[i] * s[i];

        MPI_Request req_snorm;
        double s_norm2 = 0.0;
        MPI_Iallreduce(&s_loc2, &s_norm2, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD, &req_snorm);

        /* 在等待 ||s||^2 的同时：先做 z = M^{-1} s 与本地 t_local = A*z */
        precondition_apply(z, K2_inv, s, N);
        gemv(t_local, A_local, z, N, local_rows);

        /* 等待归约结果，做收敛判断 */
        MPI_Wait(&req_snorm, MPI_STATUS_IGNORE);
        if (s_norm2 / (r0_norm2 + 1e-300) < tol_sq)
        {
            memcpy(x, h, (size_t)N * sizeof(double));
            break;
        }

        /* 收集 t 为全局（维持全量副本简化后续更新） */
        MPI_Allgatherv(t_local, local_rows, MPI_DOUBLE,
                       t, rows_counts, rows_displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        /* 合并一次归约： (t,s) 与 (t,t) */
        double ts_tt[2];
        dots2(t, s, t, t, N, ts_tt);
        double ts = ts_tt[0], tt = ts_tt[1];

        /* omega */
        double omega_try = safe_div(ts, tt, eps_den);
        if (unlikely(!isfinite(omega_try)))
        {
            if (rank == 0)
                fprintf(stderr, ">>> Breakdown: (t,t) ~ 0 at iter %d, soft-restart\n", iter);
            memcpy(p, r, (size_t)N * sizeof(double));
            continue;
        }
        omega = omega_try;

        /* x = h + omega z */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            x[i] = h[i] + omega * z[i];

        /* r = s - omega t */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            r[i] = s[i] - omega * t[i];

        /* ---------- 周期性重算 r = b - A x（仅本地块更新，去掉 Allgatherv） ---------- */
        if (unlikely(iter % k_recomp == 0))
        {
            gemv(Ax_local, A_local, x, N, local_rows);
#pragma omp parallel for schedule(static)
            for (int li = 0; li < locN; ++li)
            {
                int i = s_idx + li;
                r[i] = b[i] - Ax_local[li];
            }
        }

        /* 合并一次归约： ||r||^2 与 (r_hat, r) */
        double r2_rho[2];
        dots2(r, r, r_hat, r, N, r2_rho);
        double r_norm2 = r2_rho[0];
        rho_old = rho;
        rho = r2_rho[1];

        if (r_norm2 / (r0_norm2 + 1e-300) < tol_sq)
            break;

        if (iter % 100 == 0 || iter == 1)
        {
            if (rank == 0)
            {
                printf("[Iter %d] Residual Norm^2 = %.6e (rel=%.3e)\n",
                       iter, r_norm2, r_norm2 / (r0_norm2 + 1e-300));
                fflush(stdout);
            }
            if (!isfinite(r_norm2))
            {
                if (rank == 0)
                    fprintf(stderr, "!!! Residual became NaN/Inf at iteration %d, exiting early !!!\n", iter);
                break;
            }
        }

        if (unlikely(!isfinite(omega) || fabs(omega) < eps_den))
        {
            if (rank == 0)
                fprintf(stderr, ">>> omega too small/non-finite at iter %d, soft-restart\n", iter);
            memcpy(p, r, (size_t)N * sizeof(double));
            continue;
        }

        beta = (rho / (r0_norm2 > 0 ? (rho_old + 1e-300) : (rho_old + 1e-300))) * (alpha / omega);
        if (unlikely(!isfinite(beta)))
        {
            if (rank == 0)
                fprintf(stderr, ">>> beta non-finite at iter %d, soft-restart\n", iter);
            memcpy(p, r, (size_t)N * sizeof(double));
            continue;
        }

        /* p = r + beta * (p - omega * v) */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
    }

    /* 资源释放 */
    free(Ax_local);
    free(A_local);
    free(v_local);
    free(t_local);
    free(r);
    free(r_hat);
    free(p);
    free(v);
    free(s);
    free(h);
    free(t);
    free(y);
    free(z);
    free(K2_inv);
    if (b_alloc_local)
        free(b);
    if (x_alloc_local)
        free(x);
    free(rows_counts);
    free(rows_displs);
    free(A_counts);
    free(A_displs);

    return (iter > max_iter) ? -1 : iter;
}
