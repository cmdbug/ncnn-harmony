#ifndef BENCHMARK_NCNN_H
#define BENCHMARK_NCNN_H

#include "c_api.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"
#include "benchmark.h"
#include <string>

namespace benchmark {

typedef struct BenchmarkResult {
    int loop;
    int min;
    int max;
    int avg;
    int width;
    int height;
} BenchmarkResult;

class DataReaderFromEmpty : public ncnn::DataReader {
public:
    virtual int scan(const char *format, void *p) const;
    virtual size_t read(void *buf, size_t size) const;
};

class BenchmarkNet : public ncnn::Net {
public:
    BenchmarkResult run(int loops, double &time_min, double &time_max, double &time_avg, int &width, int &height, int size);

};


} // namespace benchmark


#endif
