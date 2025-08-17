#include <cstdlib>
#include <iostream>
#include <chrono> // 新增：计时
#include "judger.h"
#include <mpi.h>

extern "C" int bicgstab(int N, double *A, double *b, double *x, int max_iter, double tol);

int main(int argc, char *argv[])
{
    int world_rank = 0, world_size = 1;

    // 用线程友好的方式初始化 MPI：只有主线程做 MPI 调用（FUNNELED）
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        // 环境不支持 FUNNELED 时，仍可运行，但建议提示一下
        if (world_rank == 0)
            std::cerr << "[WARN] MPI does not provide MPI_THREAD_FUNNELED, provided=" << provided << "\n";
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2)
    {
        if (world_rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <input_data>\n";
        }
        MPI_Finalize();
        return -1;
    }

    std::string filename = argv[1];

    int N = 0;
    double *A = nullptr, *b = nullptr, *x = nullptr;

    // 只在 rank 0 读入
    if (world_rank == 0)
    {
        read_data(filename, &N, &A, &b, &x);
    }

    // 广播规模 N（其余数据的广播/分发在 solver.c 内部处理）
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = bicgstab(N, A, b, x, MAX_ITER, TOL);
    auto end = std::chrono::high_resolution_clock::now();

    if (world_rank == 0)
    {
        auto duration = end - start;
        judge(iter, duration, N, A, b, x);
        // 只释放 rank 0 从 read_data 得到的内存
        free(A);
        free(b);
        free(x);
    }

    MPI_Finalize();
    return 0;
}
