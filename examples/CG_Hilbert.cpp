//
// Created by creeper on 23-5-4.
//
#include <Solver.h>
using namespace nlax;

GaussSeidelSolver GS;
CGSolver cg;

uint N = 200;

Matrix buildMat()
{
    Matrix A(N, N);
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            A[i][j] = 1.0 / (i + j + 1.0);
    return A;
}

Vector buildVec(const Matrix& A)
{
    Vector b(N);
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            b[i] += A(i, j);
        b[i] /= 3.0;
    }
    return b;
}

Real sol_error(const Vector& sol, const Matrix& A, const Vector& b)
{
    return (A * sol - b).L1Norm();
}

int main()
{
    std::printf("Comparing classical solvers and conjugate gradient solver on Hilbert matrix\n");
    std::printf("The matrix is 200 * 200.\n\n");
    Matrix A = buildMat();
    Vector b = buildVec(A);
    std::printf("1. Iterate for 20 rounds and compare the result.\n");
    GS.setMaxRound(20);
    cg.setMaxRound(20);
    Vector gs_res = Vector::randVec(N);
    GS.solve(A, b, gs_res);
    Vector cg_res = Vector::randVec(N);
    cg.solve(A, b, cg_res);
    std::printf("Solution errors:\n Gauss-Seidel: %lf\n CG: %lf\n\n", sol_error(gs_res, A, b), sol_error(cg_res, A, b));
    std::printf("2. Iterate to a precision of 1e-5 and compare the rounds taken.\n");
    GS.setMaxRound(-1);
    cg.setMaxRound(-1);
    GS.setEpsilon(1e-5);
    cg.setEpsilon(1e-5);
    gs_res.randFill();
    GS.solve(A, b, gs_res);
    cg_res.randFill();
    cg.solve(A, b, cg_res);
    std::printf("Rounds:\n Gauss-Seidel: %d\n CG: %d\n", GS.rounds(), cg.rounds());
    std::printf("Solution errors:\n Gauss-Seidel: %lf\n CG: %lf\n\n", sol_error(gs_res, A, b), sol_error(cg_res, A, b));
    std::printf("Now test CG solver on 2048 * 2048 Hilbert matrix.\n");
    N = 2048;
    cg_res = Vector::randVec(N);
    A = buildMat();
    b = buildVec(A);
    printf("Start timing...\n");
    clock_t start = clock();
    cg.solve(A, b, cg_res);
    clock_t end = clock();
    double solve_time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000;
    printf("Stop timing. Taking %lf ms, %d rounds.\n The solution error is %lf\n", solve_time, cg.rounds(), sol_error(cg_res, A, b));
}