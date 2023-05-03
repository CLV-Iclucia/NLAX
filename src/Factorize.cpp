//
// Created by creeper on 22-11-5.
//
#include "Factorize.h"

namespace nlax
{
    void LUDecompose(const Matrix A, Matrix L, Matrix& M, Matrix& U)
    {
        assert(A.rows() == L.rows() && A.cols() == U.cols());
        M = A;
        L.MakeIdentity();
        U.MakeIdentity();
        uint start_up = 0, rk = 0;
        for(int i = 0; i < A.cols(); i++)
        {
            bool fullZero = true;
            for(int j = 0; j < A.rows(); j++)
            {
                if(A[j][i] != 0)
                {
                    fullZero = false;
                    break;
                }
            }
            if(!fullZero) break;
            start_up++;
        }
        for(uint i = 0, p = start_up; i < A.rows() && p < A.cols(); i++, p++, rk++)
        {
            Real max_val = std::abs(M[i][p]);
            uint idx = i;
            for(uint j = i + 1; j < A.rows(); j++)
            {
                if(std::abs(M[j][p]) > max_val)
                {
                    max_val =std::abs(M[j][p]);
                    idx = j;
                }
            }
            FirstElementaryRowOpt(M, i, idx, p, A.cols());
            FirstElementaryRowOpt(L, i, idx);
            Real NegInv = -1.0 / M[i][p];
            for(uint j = i + 1; j < A.rows(); j++)
            {
                SecondElementaryRowOpt(L, i, NegInv * M[j][p], j);
                SecondElementaryRowOpt(M, i, NegInv * M[j][p], j, p, A.cols());
            }
        }// after this, we get L and M turns into an upper triangle matrix, perform row op to calc U
        for(uint i = 0, p = start_up; i < rk; i++, p++)
        {
            Real NegInv = -1.0 / M[i][p];
            for(uint j = p + 1; j < A.cols(); j++)
            {
                SecondElementaryColumnOpt(U, p, NegInv * M[i][j], j);
                SecondElementaryColumnOpt(M, p, NegInv * M[i][j], j, i, i + 1);
            }
            ThirdElementaryColumnOpt(M, i, -NegInv, p, p + 1);
            ThirdElementaryColumnOpt(U, i, -NegInv);
        }
    }
    ///< only symmetric matrix can be diagonalized
    void diagonalize(const Matrix<Real>& A, Matrix<Real>& P, Matrix<Real>& D, bool sortEigen)
    {
        if(!isSymmetric(A)) return ;
        P.MakeIdentity();
        D.copyFrom(A);//now we perform row and column operations symmetrically and alternatively
        uint n = A.rows();
        for(int i = 0; i < n; i++)//now we are considering sub-matrix D[i ~ n - 1][i ~ n - 1]
        {
            int idx = i;//idx denotes the row whose ith element has the greatest abs value
            Real maxv = fabs(D[i][i]);
            for(int j = i + 1; j < n; j++)
                if(fabs(D[i][i]) > maxv)
                {
                    maxv = fabs(D[j][j]);
                    idx = j;//maybe more cache friendly
                }
            if(idx != i)
            {
                FirstElementaryRowOpt(D, i, idx, i, n);
                FirstElementaryColumnOpt(D, i, idx, i, n);
                FirstElementaryColumnOpt(P, i, idx);
            }
            if(EqualZero(D[i][i])) continue;
            Real NegInv = -1.0 / D[i][i];
            for(int j = i + 1; j < n ; j++)
            {
                Real k = NegInv * D[i][j];
                if(!EqualZero(k))
                {
                    SecondElementaryRowOpt(D, i, k, j, i, n);
                    SecondElementaryColumnOpt(D, i, k, j, i, n);
                    SecondElementaryColumnOpt(P, j, -k, i);
                }
            }
        }
        if(sortEigen)
        {
            for(int i = 0; i < n - 1; i++)
            {
                int idx = i;
                Real maxv = fabs(D[i][i]);
                for(int j = i + 1; j < n; j++)
                {
                    if(fabs(D[j][j]) > maxv)
                    {
                        maxv = fabs(D[idx][idx]);
                        idx = j;
                    }
                }
                if(i != idx)std::swap(D[i][i], D[idx][idx]);
                FirstElementaryColumnOpt(P, i, idx);
            }
        }
    }

    void QRDecompose(const Matrix<Real>& A, Matrix<Real>& Q, Matrix<Real>& R)
    {
        assert(A.rows() == A.cols());
        int n = A.cols();
        Vec* C = A.ColumnVector();
        Vec* q = new Vec[n];
        for(int i = 0; i < n; i++)
            q[i].resize(n);
        GramSchmidt(C, q, n);
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                Q[i][j] = q[j][i];
        R = Q.transpose() * A;
    }
}