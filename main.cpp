#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <mpi.h>

#include "MatrixVector.h"

Matrix create_test_matrix(int N){
    Matrix result(N, N);
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            result.at(i, j) = 1.0 / ((i-j)*(i-j) - 0.25);
        }
    }
    return result;
}

int main (int argc, char* argv[])
{
    int world_size;
    int world_rank;
    int world_root = 0;  // root-process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Data initialization */
    int     N = 3001;
    Matrix  A;
    Vector  rhs;
    Vector  exact_sol;
    Vector  x;

    int *block_size        = new int[world_size];
    int *block_index       = new int[world_size];
    int *block_matrix_size = new int[world_size];
    int *block_matrix_index= new int[world_size];

    DefineSplittingParameters(block_index, block_size, N, MPI_COMM_WORLD);
    for (int i = 0; i < world_size; ++i){
        block_matrix_size[i] = block_size[i] * N;
        block_matrix_index[i] = block_index[i] * N;
    }

    if (world_rank == world_root){
        A = create_test_matrix(N);
        exact_sol.Fill(N, 1.0);
        rhs = A * exact_sol;
        x.Fill(N, 0.0);          // Initial approximation
    }

    Matrix A_block(block_size[world_rank], N);
    Vector x_block(block_size[world_rank]);

    MatrixScatter(A, A_block, block_matrix_size, block_matrix_index, world_root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = - MPI_Wtime();

    /* BiCGStab */
    int    max_iterations = N;
    int    iterations     = 0;
    double error;
    double rel_error;
    double tolerance      = 10*std::numeric_limits<double>::epsilon();
    bool   next_iteration = true;

    double rho;
    double alpha;
    double omega;
    double epsilon;

    Vector r;
    Vector r0;
    Vector v;
    Vector t;
    Vector p(N);
    Vector s(N);

    if (world_rank == world_root){
        rho   = 1.0;
        alpha = 1.0;
        omega = 1.0;

        r  = rhs - A * x;
        r0 = r;

        v.Fill(N, 0.0);
        t.Fill(N, 0.0);

        epsilon = tolerance * tolerance * DotProduct(rhs, rhs);
        if (DotProduct(r, r) < epsilon || iterations > max_iterations){
            next_iteration = false;
        }
    }
    MPI_Bcast(&next_iteration, 1, MPI_CXX_BOOL, world_root, MPI_COMM_WORLD);

    while (next_iteration){
        ++iterations;

        if (world_rank == world_root){
            double rho_old = rho;
            rho = DotProduct(r0, r);
            double beta = (rho / rho_old) * (alpha / omega);
            p = r + beta * (p - omega * v);
        }
        MPI_Bcast(p.GetData(), N, MPI_DOUBLE, world_root, MPI_COMM_WORLD);
        x_block = A_block * p;
        VectorGather(v, x_block, block_size, block_index, world_root, MPI_COMM_WORLD);

        if (world_rank == world_root){
            alpha = rho / DotProduct(r0, v);
            s = r - alpha * v;
        }
        MPI_Bcast(s.GetData(), N, MPI_DOUBLE, world_root, MPI_COMM_WORLD);
        x_block = A_block * s;
        VectorGather(t, x_block, block_size, block_index, world_root, MPI_COMM_WORLD);

        if (world_rank == world_root){
            double t_sqnorm = DotProduct(t, t);
            if ( t_sqnorm > 0.0 ){
                omega = DotProduct(t, s) / t_sqnorm;
            }
            else {
                omega = 0.0;
            }
            x = x + alpha * p + omega * s;
            r = s - omega * t;
            if (DotProduct(r, r) < epsilon || iterations >= max_iterations){
                next_iteration = false;
            }
        }
        MPI_Bcast(&next_iteration, 1, MPI_CXX_BOOL, world_root, MPI_COMM_WORLD);
    }

    if (world_rank == world_root){
        error = std::sqrt(DotProduct(r, r) / DotProduct(rhs, rhs));  // relative error
        rel_error = DotProduct(x - exact_sol, x - exact_sol) / N;
    }
    elapsed_time += MPI_Wtime();

    if (world_rank == world_root){
        std::cout << "--- MPI in matrix product for BiCGStab ---\n"
                  << "    Number of processors = " << world_size << '\n'
                  << "    Vector size          = " << N << '\n'
                  << "    BiCGStab iterations  = " << iterations << '\n'
                  << "    BiCGStab rel. error  = " << error << '\n'
                  << "    error (l2-norm)      = " << rel_error << '\n'
                  << "    BiCGStab time        = " << elapsed_time << " s.\n"
                  << "---------------------------------------------\n";

        std::string filename = "output/MPI_proc_" + std::to_string(world_size) +
                               "_N_" + std::to_string(N) + ".txt";
        std::ofstream out(filename);
        out << "--- MPI in matrix product for BiCGStab ---\n"
            << "    Number of processors = " << world_size << '\n'
            << "    Vector size          = " << N << '\n'
            << "    BiCGStab iterations  = " << iterations << '\n'
            << "    BiCGStab rel. error  = " << error << '\n'
            << "    error (l2-norm)      = " << rel_error << '\n'
            << "    BiCGStab time        = " << elapsed_time << " s.\n"
            << "---------------------------------------------\n";
    }
    MPI_Finalize();
    return 0;
}
