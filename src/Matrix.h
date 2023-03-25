#ifndef XMATH_MATRIX_H
#define XMATH_MATRIX_H
#include "NLAX_Vector.h"
#include <cstring>
#include <cassert>
#include "xmath_common.h"

namespace xmath
{
    template<typename T>
    class Matrix
    {
        private:
            uint m, n;//m: the number of rows. n: the number of columns
            T* a = nullptr;//we use row-major order, and we only support left multiplication to a vector
        public:
            Matrix() { m = n = 0; }
            Matrix(uint _m, uint _n) : m(_m), n(_n)
            {
                static_assert(std::is_arithmetic<T>(), "Error: Elements of matrix must be arithmetic type!");
                a = new T[m * n];
                memset(a, static_cast<T>(0), sizeof(T) * m * n);
            }
            Matrix(const Matrix& A) : m(A.m), n(A.n)
            {
                a = new T[m * n];
                for(int i = 0; i < m * n; i++)
                    a[i] = A.a[i];
            }
            Matrix(Matrix&& A) noexcept
            {
                m = std::move(A.m);
                n = std::move(A.n);
                delete[] a;
                a = std::move(A.a);
            }
            Matrix& operator=(const Matrix& A)
            {
                if(&A == this) return *this;
                if(m * n != A.m * A.n)
                {
                    delete[] a;
                    a = new T[A.m * A.n];
                }
                m = A.m;
                n = A.n;
                for(int i = 0; i < m * n; i++)
                    a[i] = A.a[i];
                return *this;
            }
            Matrix& operator=(Matrix&& A) noexcept
            {
                m = std::move(A.m);
                n = std::move(A.n);
                delete[] a;
                a = std::move(A.a);
                return *this;
            }
            uint getM() const { return m; }
            uint getN() const { return n; }
            T* operator[](uint i) { return a + i * n; }
            const T* operator[](uint i) const { return a + i * n; }
            Vector<T> operator*(const Vector<T>& v) const
            {
                assert(n == v.dim());
                Vector<T> ret(m);
                for(int i = 0; i < m; i++)
                {
                    T* Ar = a + i * n;
                    for(int j = 0; j < n; j++)
                        ret[i] += Ar[j] * v[j];
                }
                return ret;
            }
            Matrix operator*(const Matrix& A) const
            {
                assert(n == A.m);
                Matrix ret(m, A.n);
                for(int i = 0 ; i < m; i++)
                    for(int k = 0; k < n; k++)
                        for(int j = 0; j < A.n; j++)//maybe this can make use of the cache?
                            ret[i][j] += a[i * n + k] * A[k][j];
                return ret;
            }
            Matrix operator+(const Matrix& A) const
            {
                Matrix ret(m, n);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        ret[i][j] = a[i * n + j] + A[i][j];
                return ret;
            }
            Matrix operator-(const Matrix& A) const
            {
                Matrix ret(m, n);
                for (int i = 0; i < m; i++)
                {
                    T* ar = a + i * n;
                    for (int j = 0; j < n; j++)
                        ret[i][j] = ar[j] - A[i][j];
                }
                return ret;
            }
            void ColumnVector(Vector<T>* C) const
            {
                for(int i = 0; i < n; i++)
                    for(int j = 0; j < m; j++)
                        C[i][j] = *(a + n * j + i);
            }
            Vector<T>* ColumnVector() const
            {
                Vector<T>* C = new Vector<T>[n];
                for(int i = 0; i < n; i++)
                    C[i].resize(n);
                for(int i = 0; i < n; i++)
                    for(int j = 0; j < m; j++)
                        C[i][j] = *(a + n * j + i);
                return C;
            }
            void output() const
            {
                for(int i = 0; i < m; i++)
                {
                    for(int j = 0; j < n; j++)
                    {
                        if(!EqualZero(a[i * n] + j))std::cout << a[i * n + j] << ",";
                        else std::cout << 0 << ",";
                    }
                    std::cout << std::endl;
                }
                std::cout << "-----------" << std::endl;
            }
            void MakeIdentity()
            {
                memset(a, 0, sizeof(T) * m * n);
                for(int i = 0; i < std::min(m, n); i++)
                    a[i * n + i] = static_cast<T>(1);
            }
            void copyFrom(const Matrix& A)
            {
                for(int i = 0; i < A.m; i++)
                    for(int j = 0; j < A.n; j++)
                        a[i * n + j] = A[i][j];
            }
            Matrix transpose() const
            {
                Matrix ret(n, m);
                for(int i = 0; i < m; i++)
                    for(int j = 0; j < n; j++)
                        ret[j][i] = a[i * n + j];
                return ret;
            }
            bool isSquare() const { return m == n; }
            ~Matrix() { delete[] a; }
    };
    template class Matrix<Real>;
    template class Matrix<int>;
    template<typename T1, typename T2> Matrix<T2> matrix_cast(const Matrix<T1>& M);
    void FirstElementaryRowOpt(Matrix<Real>&, uint, uint);
    void FirstElementaryColumnOpt(Matrix<Real>&, uint, uint);
    void SecondElementaryRowOpt(Matrix<Real>&, uint, Real, uint);
    void SecondElementaryColumnOpt(Matrix<Real>&, uint, Real, uint);
    void ThirdElementaryRowOpt(Matrix<Real>&, uint, Real);
    void ThirdElementaryColumnOpt(Matrix<Real>&, uint, Real);
    void FirstElementaryRowOpt(Matrix<Real>&, uint, uint, uint, uint);
    void FirstElementaryColumnOpt(Matrix<Real>&, uint, uint, uint, uint);
    void SecondElementaryRowOpt(Matrix<Real>&, uint, Real, uint, uint, uint);
    void SecondElementaryColumnOpt(Matrix<Real>&, uint, Real, uint, uint, uint);
    void ThirdElementaryRowOpt(Matrix<Real>&, uint, Real, uint, uint);
    void ThirdElementaryColumnOpt(Matrix<Real>&, uint, Real, uint, uint);
    bool isSymmetric(const Matrix<Real>&);
}
#endif //XMATH_MATRIX_H
