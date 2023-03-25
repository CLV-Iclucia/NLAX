//
// Created by creeper on 22-11-5.
//

#ifndef XMATH_FACTORIZE_H
#define XMATH_FACTORIZE_H
#include "NLAX_Vector.h"
#include "Matrix.h"
namespace nlax
{
    void LUFactorize(const Matrix<Real>&, Matrix<Real>&, Matrix<Real>&, Matrix<Real>&);
    void QRFactorize(const Matrix<Real>&, Matrix<Real>&, Matrix<Real>&);
    void diagonalize(const Matrix<Real>&, Matrix<Real>&, Matrix<Real>&, bool sortEigen = true);//diag
}
#endif //XMATH_FACTORIZE_H
