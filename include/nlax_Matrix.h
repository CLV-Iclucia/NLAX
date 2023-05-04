#ifndef NLAX_MATRIX_H
#define NLAX_MATRIX_H
#include "nlax_Vector.h"

#include <cstring>
#include <cassert>
#include <iostream>

namespace nlax
{
    class Matrix
    {
        private:
            uint m, n;//m: the number of rows. n: the number of columns
            Real *a = nullptr;//we use row-major order, and we only support left multiplication to a vector
        public:
            Matrix()
            { m = n = 0; }

            Matrix(uint _m, uint _n) : m(_m), n(_n)
            {
                a = new Real[m * n];
                memset(a, 0, sizeof(Real) * m * n);
            }

            Matrix(const Matrix &A) : m(A.m), n(A.n)
            {
                a = new Real[m * n];
                memcpy(a, A.a, sizeof(Real) * m * n);
            }

            Matrix(Matrix &&A) noexcept
            {
                m = A.m;
                n = A.n;
                delete[] a;
                a = A.a;
                A.a = nullptr;
            }

            Matrix &operator=(const Matrix &A)
            {
                if (&A == this) return *this;
                if (m * n != A.m * A.n)
                {
                    delete[] a;
                    a = new Real[A.m * A.n];
                }
                m = A.m;
                n = A.n;
                memcpy(a, A.a, m * n * sizeof(Real));
                return *this;
            }

            Matrix &operator=(Matrix &&A) noexcept
            {
                if (&A == this) return *this;
                m = A.m;
                n = A.n;
                delete[] a;
                a = A.a;
                A.a = nullptr;
                return *this;
            }

            uint rows() const
            { return m; }

            uint cols() const
            { return n; }

            Real *operator[](uint i)
            { return a + i * n; }

            const Real *operator[](uint i) const
            { return a + i * n; }

            Real operator()(uint i, uint j) const
            { return *(a + i * n + j); }

            Real &operator()(uint i, uint j)
            { return *(a + i * n + j); }

            Vector operator*(const Vector &v) const
            {
                assert(n == v.dim());
                Vector ret(m);
                for (int i = 0; i < m; i++)
                {
                    Real *Ar = a + i * n;
                    for (int j = 0; j < n; j++)
                        ret[i] += Ar[j] * v[j];
                }
                return ret;
            }

            Matrix operator*(const Matrix &A) const
            {
                assert(n == A.m);
                Matrix ret(m, A.n);
                for (int i = 0; i < m; i++)
                    for (int k = 0; k < n; k++)
                        for (int j = 0; j < A.n; j++)
                            ret[i][j] += a[i * n + k] * A[k][j];
                return ret;
            }

            Matrix operator+(const Matrix &A) const
            {
                Matrix ret(m, n);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        ret[i][j] = a[i * n + j] + A[i][j];
                return ret;
            }

            Matrix operator-(const Matrix &A) const
            {
                Matrix ret(m, n);
                for (int i = 0; i < m; i++)
                {
                    Real *ar = a + i * n;
                    for (int j = 0; j < n; j++)
                        ret[i][j] = ar[j] - A[i][j];
                }
                return ret;
            }

            void ColumnVector(Vector *C) const
            {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++)
                        C[i][j] = *(a + n * j + i);
            }

            friend std::ostream &operator<<(std::ostream &o, const Matrix &mat)
            {
                for (uint i = 0; i < mat.rows(); i++, o << std::endl)
                    for (uint j = 0; j < mat.cols(); j++)
                        o << mat(i, j) << " ";
            }

            void MakeIdentity()
            {
                memset(a, 0, sizeof(Real) * m * n);
                for (int i = 0; i < std::min(m, n); i++)
                    a[i * n + i] = 1.0;
            }

            void copyFrom(const Matrix &A)
            {
                for (int i = 0; i < A.m; i++)
                    for (int j = 0; j < A.n; j++)
                        a[i * n + j] = A[i][j];
            }

            Matrix transpose() const
            {
                Matrix ret(n, m);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        ret[j][i] = a[i * n + j];
                return ret;
            }

            Vector row(uint i) const
            { return Vector(n, a + i * n); }

            bool isSquare() const
            { return m == n; }

            bool isSymmetric() const
            {
                if (!isSquare()) return false;
                for (uint i = 0; i < m; i++)
                    for (int j = 0; j < i; j++)
                        if (!isEqual(a[i * n + j], a[j * n + i])) return false;
                return true;
            }

            Vector diag() const
            {
                assert(m == n);
                Vector ret(m);
                for(int i = 0; i < m; i++)
                    ret[i] = a[i * n + i];
                return ret;
            }

            ~Matrix()
            { delete[] a; }
    };
}


#endif //XMATH_MATRIX_H
