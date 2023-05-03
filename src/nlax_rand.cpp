//
// Created by creeper on 23-3-26.
//
#include <nlax_rand.h>
namespace nlax
{
    static std::uniform_real_distribution<Real> e(0, 1);
    Real get_random() { return std::distrib(e); }
    void random_fill(Vector& v)
    {
        for(uint i = 0; i < v.dim(); i++)
            v[i] = get_random();
    }
}