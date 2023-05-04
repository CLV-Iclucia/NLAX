//
// Created by creeper on 23-3-25.
//

#ifndef NLAX_SOLVER_H
#define NLAX_SOLVER_H
#include <nlax_types.h>
#include <nlax_utils.h>
#include "nlax_Matrix.h"
#include "nlax_Vector.h"

namespace nlax
{
    enum SolverStatus { Undefined, Success, NumericalError, NotConverge };
    class Solver
    {
        protected:
            SolverStatus status = Undefined;
        public:
            SolverStatus info() const { return status; }

    };
    class FactorSolver : public Solver
    {
        public:
            virtual void solve(const Vector& b, Vector& ret) = 0;
            Vector solve(const Vector& b)
            {
                Vector ret;
                solve(b, ret);
                return ret;
            };
    };

    class IterativeSolver : public Solver
    {
        protected:
            int maxRound = -1;
            Real conv_eps = EPS;
            uint nRounds = 0;
        public:
            void setEpsilon(Real eps) { conv_eps = eps; }
            void setMaxRound(int _maxRound) { maxRound = _maxRound < 0 ? -1 : _maxRound; }
            uint rounds() const { return nRounds; }
            virtual void solve(const Matrix& A, const Vector& b, Vector& ret) = 0;
            virtual Vector solve(const Matrix& A, const Vector& b)
            {
                Vector ret;
                solve(A, b, ret);
                return ret;
            }
    };

    class JacobiSolver final : public IterativeSolver
    {
        public:
            void solve(const Matrix& A, const Vector& b, Vector& ret) override
            {
                assert(A.isSquare() && A.cols() == b.dim() && ret.dim() == b.dim());
                uint n = b.dim();
                Vector Dinv = A.diag().inv();
                Vector last;
                for(nRounds = 1; maxRound < 0 || nRounds < maxRound; nRounds++)
                {
                    last = ret;
                    for(int i = 0; i < n; i++)
                        ret[i] = (-A.row(i).dot(last) + b[i]) * Dinv[i] + last[i];
                    if((ret - last).L1Norm() < conv_eps) break;
                }
            }
    };

    class GaussSeidelSolver final : public IterativeSolver
    {
        public:
            void solve(const Matrix& A, const Vector& b, Vector& ret) override
            {
                assert(A.isSquare() && A.cols() == b.dim() && ret.dim() == b.dim());
                uint n = b.dim();
                Vector Dinv = A.diag().inv();
                Vector last;
                for(nRounds = 1; maxRound < 0 || nRounds < maxRound; nRounds++)
                {
                    last = ret;
                    for(int i = 0; i < n; i++)
                    {
                        ret[i] = b[i];
                        for(int j = 0; j < i; j++)
                            ret[i] -= A(i, j) * ret[j];
                        for(int j = i + 1; j < n; j++)
                            ret[i] -= A(i, j) * last[j];
                        ret[i] *= Dinv[i];
                    }
                    if((ret - last).L1Norm() < conv_eps) break;
                }
            }
    };

    class SORSolver final: public IterativeSolver
    {
        private:
            Real w = 0.5;
        public:
            SORSolver() = default;
            void setw(Real _w) { w = _w; }
            void solve(const Matrix& A, const Vector& b, Vector& ret) override
            {
                assert(A.isSquare() && A.cols() == b.dim() && ret.dim() == b.dim());
                uint n = b.dim();
                Vector Dinv = A.diag().inv();
                Vector last;
                for (nRounds = 1; maxRound < 0 || nRounds < maxRound; nRounds++)
                {
                    last = ret;
                    for (int i = 0; i < n; i++)
                    {
                        ret[i] = w * Dinv[i] * b[i] + (1.0 - w) * last[i];
                        for (int j = 0; j < i; j++)
                            ret[i] -= w * A(i, j) * ret[j] * Dinv[i];
                        for (int j = i + 1; j < n; j++)
                            ret[i] -= w * A(i, j) * last[j] * Dinv[i];
                    }
                    if ((ret - last).L1Norm() < conv_eps) break;
                }
            }
    };

    class CGSolver : public IterativeSolver
    {
        public:
            void solve(const Matrix& A, const Vector& b, Vector& ret) override
            {
                assert(A.isSymmetric());
                assert(A.isSymmetric() && A.cols() == ret.dim());
                Vector r(b - A * ret);
                Vector p(r);
                Real alpha, beta, rNormSqrCache = r.L2NormSqr();
                Vector Ap = A * p;
                for(nRounds = 1; maxRound < 0 || nRounds < maxRound; nRounds++)
                {
                    alpha = rNormSqrCache / p.dot(Ap);
                    ret.saxpy(p, alpha);
                    if(alpha * p.L1Norm() < conv_eps) break;
                    r.saxpy(Ap, -alpha);
                    beta = r.L2NormSqr() / rNormSqrCache;
                    rNormSqrCache = r.L2NormSqr();
                    p.scadd(r, beta);
                    Ap = A * p;
                }
            }
    };
}


#endif //NLAX_SOLVER_H
