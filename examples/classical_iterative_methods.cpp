#include <Solver.h>
using namespace nlax;

JacobiSolver jacobi;
GaussSeidelSolver GS;
SORSolver sor;

const Real a = 0.5;
const Real e = 1.0;
const uint N = 100;
const Real h = 1.0 / N;

Matrix buildMat()
{
    Matrix A(N, N);
    A(0, 0) = -2.0 * e - h;
    A(0, 1) = e + h;
    for(int i = 1; i < N - 1; i++)
    {
        A(i, i - 1) = e;
        A(i, i) = -2.0 * e - h;
        A(i, i + 1) = e + h;
    }
    A(N - 1, N - 2) = e;
    A(N - 1, N - 1) = -2.0 * e - h;
    return A;
}

Vector buildVec()
{
    Vector b(N);
    b.fill(a * h * h);
    return b;
}

Real error(const Vector& sol)
{
    Real sum = 0.0;
    for(int i = 0; i < N; i++)
    {
        Real y = (1.0 - a) / (1.0 - std::exp(-1.0 / e)) * (1.0 - exp(-i * h / e)) + a * i * h;
        sum += std::abs(sol[i] - y);
    }
    return sum;
}

Real sol_error(const Vector& sol, const Matrix& A, const Vector& b)
{
    return (A * sol - b).L1Norm();
}

int main()
{
    std::printf("Comparing Jacobi solver, G-S solver and SOR solver on a tri-diagonal linear system\n");
    std::printf("The matrix is 100 * 100.\n\n");
    Matrix A = buildMat();
    Vector b = buildVec();
    std::printf("1. Iterate for 20 rounds and compare the result.\n");
    jacobi.setMaxRound(20);
    GS.setMaxRound(20);
    sor.setMaxRound(20);
    Vector initV = Vector::randVec(N);
    Vector jac_res = initV;
    jacobi.solve(A, b, jac_res);
    Vector gs_res = initV;
    GS.solve(A, b, gs_res);
    Vector sor_res = initV;
    sor.solve(A, b, sor_res);
    std::printf("Errors:\n Jacobi: %lf\n Gauss-Seidel: %lf\n SOR: %lf\n", error(jac_res), error(gs_res), error(sor_res));
    std::printf("Solution errors:\n Jacobi: %lf\n Gauss-Seidel: %lf\n SOR: %lf\n\n", sol_error(jac_res, A, b),
                sol_error(gs_res, A, b), sol_error(sor_res, A, b));
    std::printf("2. Iterate to a precision of 1e-5 and compare the rounds taken.\n");
    jacobi.setMaxRound(-1);
    GS.setMaxRound(-1);
    sor.setMaxRound(-1);
    jacobi.setEpsilon(1e-5);
    GS.setEpsilon(1e-5);
    sor.setEpsilon(1e-5);
    jac_res.randFill();
    jacobi.solve(A, b, jac_res);
    gs_res.randFill();
    GS.solve(A, b, gs_res);
    sor_res.randFill();
    sor.solve(A, b, sor_res);
    std::printf("Rounds:\n Jacobi: %d\n Gauss-Seidel: %d\n SOR: %d\n", jacobi.rounds(), GS.rounds(), sor.rounds());
    std::printf("Errors:\n Jacobi: %lf\n Gauss-Seidel: %lf\n SOR: %lf\n", error(jac_res), error(gs_res), error(sor_res));
    std::printf("Solution errors:\n Jacobi: %lf\n Gauss-Seidel: %lf\n SOR: %lf\n\n", sol_error(jac_res, A, b),
                sol_error(gs_res, A, b), sol_error(sor_res, A, b));
    std::printf("3. Compare the effect of SOR with different omegas.\n");
    for(int i = 1; i <= 19; i++)
    {
        Real w = i * 0.1;
        std::printf("(%d) omega = %lf, take ", i, w);
        sor.setw(w);
        sor_res.randFill();
        sor.solve(A, b, sor_res);
        std::printf("%d rounds\n", sor.rounds());
        std::printf(" Error: %lf Solution error: %lf\n", error(sor_res), sol_error(sor_res, A, b));
    }
}