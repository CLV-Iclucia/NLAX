//
// Created by creeper on 23-4-30.
//
#include <benchmark/benchmark.h>
#include <nlax_Vector.h>
using namespace nlax;

Vector result;

static void bench_vec_add(benchmark::State& state)
{
    Vector A = Vector::randVec();
    Vector B = Vector::randVec(A.dim());
    for(auto _ : state)
        result = A + B;
}

BENCHMARK(bench_vec_add);

BENCHMARK_MAIN();