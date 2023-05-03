//
// Created by creeper on 23-3-25.
//

#ifndef NLAX_NLAX_UTILS_H
#define NLAX_NLAX_UTILS_H

#include <nlax_types.h>
#include <random>
namespace nlax
{
    inline bool isZero(Real x) { return x >= -EPS && x <= EPS; }
    inline bool isEqual(Real x, Real y) { return x - y >= -EPS && x - y <= EPS; }
    inline uint randu()
    {
        static std::random_device seed;
        std::default_random_engine e(seed());
        static std::uniform_int_distribution<uint> u(0, 10);
        return u(e);
    }
    inline Real randReal()
    {
        static std::random_device seed;
        std::default_random_engine e(seed());
        static std::uniform_real_distribution<Real> u(0.0, 1.0);
        return u(e);
    }
}

#endif //NLAX_NLAX_UTILS_H
