#ifndef NLAX_VECTOR_H
#define NLAX_VECTOR_H
#include <utility>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <nlax_types.h>
#include <nlax_utils.h>
#include <cstring>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

// TODO: Add SIMD
#include <immintrin.h>

namespace nlax
{
    class Vector
    {
        private:
            unsigned int n = 0;
            Real* v = nullptr;
        public:
            void randFill()
            {
                for(uint i = 0; i < n; i++)
                    v[i] = randReal();
            }
            static Vector randVec()
            {
                uint _n = randu();
                Vector vec(_n);
                vec.randFill();
                return vec;
            }
            static Vector randVec(uint _n)
            {
                Vector vec(_n);
                vec.randFill();
                return vec;
            }
            Vector() = default;
            Vector(unsigned int _n, const Real* A) : n(_n)
            {
                v = new Real[n];
                memcpy(v, A, sizeof(Real) * n);
            }
            explicit Vector(unsigned int _n) : n(_n)
            {
                v = new Real[n];
                memset(v, 0, sizeof(Real) * n);
            }
            Vector(const Vector& A)
            {
                n = A.n;
                v = new Real[n];
                memcpy(v, A.v, sizeof(Real) * n);
            }
            Vector(Vector&& A) noexcept
            {
                n = A.n;
                v = A.v;
                A.v = nullptr;
            }
            Vector& operator=(const Vector& A)
            {
                if(&A == this) return *this;
                else
                {
                    if(n != A.n)
                    {
                        n = A.n;
                        delete[] v;
                        v = new Real[n];
                    }
                    memcpy(v, A.v, sizeof(Real) * n);
                    return *this;
                }
            }
            Vector& operator=(Vector&& A) noexcept
            {
                if(n != A.n)
                {
                    n = A.n;
                    delete[] v;
                    v = A.v;
                    A.v = nullptr;
                }
                else memcpy(v, A.v, sizeof(Real) * n);
                return *this;
            }
            void fill(Real val)
            {
                for(uint i = 0; i < n; i++)
                    v[i] = val;
            }
            Real dot(const Vector& A) const
            {
                Real sum = 0.0;
                int i = 0;
//#pragma omp parallel for reduction(+:sum)
                for (i = 0; i < n; i++)
                    sum += v[i] * A.v[i];
                return sum;
            }
            void resize(int _n)
            {
                n = _n;
                delete[] v;
                v = new Real[n];
            }
            Real& operator[](int i) { return v[i]; }
            const Real& operator[](int i) const { return v[i]; }
            Vector operator*(const Vector& A) const
            {
                Vector V(n);
                int i = 0;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    V.v[i] = v[i] * A.v[i];
                return V;
            }
            void saxpy(const Vector& A, Real k)
            {
                int i = 0;
//#pragma omp parallel for
                for(i = 0; i < n; i++)
                    v[i] += A.v[i] * k;
            }
            void scadd(const Vector& A, Real k)
            {
                int i = 0;
//#pragma omp parallel for
                for(i = 0; i < n; i++)
                    v[i] = k * v[i] + A.v[i];
            }
            Vector operator+(const Vector& A) const
            {
                Vector V(n);
                int i = 0;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    V.v[i] = v[i] + A.v[i];
                return V;
            }
            Vector operator-()
            {
                Vector V(n);
                int i = 0;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    V.v[i] = -v[i];
                return V;
            }
            Vector operator/(const Vector& A) const
            {
                Vector V(n);
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    V.v[i] = v[i] / A.v[i];
                return V;
            }
            Vector operator-(const Vector& A) const
            {
                Vector V(n);
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    V.v[i] = v[i] - A.v[i];
                return V;
            }
            Vector operator/(Real val) const
            {
                if (val == 0)
                {
                    std::printf("Division by zero!");
                    exit(-1);
                }
                Vector V(n);
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    V.v[i] = v[i] / val;
                return V;
            }
            Vector operator*(Real val) const
            {
                Vector V(n);
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    V.v[i] = val * v[i];
                return V;
            }
            friend Vector operator*(Real val, const Vector& A)
            { return A * val; }
            Vector& operator*=(const Vector& A)
            {
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    v[i] *= A.v[i];
                return *this;
            }
            Vector& inv()
            {
                for(uint i = 0; i < n; i++)
                    v[i] = 1.0 / v[i];
                return *this;
            }
            Vector inv() const
            {
                Vector ret;
                for(uint i = 0; i < n; i++)
                    ret[i] = 1.0 / v[i];
                return ret;
            }
            Vector& operator+=(const Vector& A)
            {
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    v[i] += A.v[i];
                return *this;
            }
            Vector& operator/=(const Vector& A)
            {
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    v[i] /= A.v[i];
                return *this;
            }
            Vector& operator/=(Real val)
            {
                if (val == 0)
                {
                    std::printf("Division by zero!");
                    exit(-1);
                }
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    v[i] /= val;
                return *this;
            }
            Vector& operator-=(const Vector& A)
            {
                int i;
//#pragma omp parallel for
                for (i = 0; i < n; i++)
                    v[i] -= A.v[i];
                return *this;
            }
            Vector& operator*=(Real val)
            {
                for (int i = 0; i < n; i++)
                    v[i] *= val;
                return *this;
            }
            uint dim() const { return n; }
            Real L2Norm() const
            {
                assert(n > 0);
                Real maxv = std::abs(v[0]), sum = 0.0;
                for(uint i = 1; i < n; i++)
                    maxv = std::max(std::abs(v[i]), maxv);
                for(uint i = 0; i < n; i++)
                    sum += (v[i] / maxv) * (v[i] / maxv);
                return std::sqrt(sum) * maxv;
            }
            Real L2NormSqr() const
            {
                assert(n > 0);
                Real maxv = std::abs(v[0]), sum = 0.0;
                for(uint i = 1; i < n; i++)
                    maxv = std::max(std::abs(v[i]), maxv);
                for(uint i = 0; i < n; i++)
                    sum += (v[i] / maxv) * (v[i] / maxv);
                return sum * maxv * maxv;
            }
            Real L1Norm() const
            {
                assert(n > 0);
                Real sum = 0.0;
                for(uint i = 0; i < n; i++)
                    sum += std::abs(v[i]);
                return sum;
            }
            Vector normalized() const
            {
                Real norm = L2Norm();
                Vector ret(n);
                for(int i = 0; i < n; i++)
                    ret[i] = v[i] / norm;
                return ret;
            }
            friend std::ostream& operator<<(std::ostream& o, const Vector& A)
            {
                o << A.v[0];
                for(uint i = 1; i < A.n; i++)
                    o << " " << A.v[i];
                return o;
            }
            ~Vector() { delete[] v; }
    };

    Real dot(const Vector& A, const Vector& B)
    {
        assert(A.dim() == B.dim());
        Real ret = 0.0;
        for(int i = 0; i < A.dim(); i++)
            ret += A[i] * B[i];
        return ret;
    }

    Real norm(const Vector& A) { return std::sqrt(dot(A, A)); }
    inline void normalize(Vector& v)
    {
        Real len = norm(v);
        if(isZero(len)) return ;
        v /= len;
    }
}

#endif //XMATH_TVECTOR_H