#include "benchmark_ncnn.h"
#include <cfloat>
#include <map>

#include "hilog/log.h"

#undef LOG_TAG
#define LOG_TAG "Tncnn"

namespace benchmark {


int benchmark::DataReaderFromEmpty::scan(const char *format, void *p) const { return 0; }

size_t benchmark::DataReaderFromEmpty::read(void *buf, size_t size) const {
    memset(buf, 0, size);
    return size;
}


benchmark::BenchmarkResult benchmark::BenchmarkNet::run(int loops, double &time_min, double &time_max, double &time_avg, int &width,
                                 int &height, int size) {
    time_min = DBL_MAX;
    time_max = -DBL_MAX;
    time_avg = 0;

    // resolve input shape
    const std::vector<const char *> &input_names_x = input_names();
    const std::vector<const char *> &output_names_x = output_names();
    std::vector<ncnn::Mat> input_mats_x;

    {
        for (int i = 0; i < (int)layers().size(); i++) {
            const ncnn::Layer *layer = layers()[i];

            if (layer->type != "Input") {
                continue;
            }

            for (int j = 0; j < input_names_x.size(); j++) {
                if (blobs()[layer->tops[0]].name != input_names_x[j]) {
                    continue;
                }

                ncnn::Mat in;
                const ncnn::Mat &shape = layer->top_shapes[0];
                if (shape.c == 0 || shape.h == 0 || shape.w == 0) {
                    in.create(size, size, 3);
                    width = size;
                    height = size;
                } else {
                    in.create(shape.w, shape.h, shape.c);
                    width = shape.w;
                    height = shape.h;
                }
                in.fill(0.01f);
                input_mats_x.push_back(in);
            }
        }

        if (input_names_x.empty()) {
            BenchmarkResult benchmarkResult;
            return benchmarkResult;
        }
        if (input_mats_x.empty()) {
            BenchmarkResult benchmarkResult;
            return benchmarkResult;
        }
        if (input_names_x.size() != input_mats_x.size()) {
            BenchmarkResult benchmarkResult;
            return benchmarkResult;
        }
    }

    ncnn::Mat out;

    // warm up
    const int g_warmup_loop_count = 5; // FIXME hardcode
    for (int i = 0; i < g_warmup_loop_count; i++) {
//         OH_LOG_DEBUG(LogType::LOG_APP, "warm up:%{public}d", i + 1);

        ncnn::Extractor ex = create_extractor();
        for (int j = 0; j < input_names_x.size(); j++) {
            ex.input(input_names_x[j], input_mats_x[j]);
        }
        for (int j = 0; j < output_names_x.size(); j++) {
            ex.extract(output_names_x[j], out);
        }
    }

    for (int i = 0; i < loops; i++) {
//         OH_LOG_DEBUG(LogType::LOG_APP, "benchmark run:%{public}d", i + 1);

        double start = ncnn::get_current_time();
        {
            ncnn::Extractor ex = create_extractor();
            for (int j = 0; j < input_names_x.size(); j++) {
                ex.input(input_names_x[j], input_mats_x[j]);
            }
            for (int j = 0; j < output_names_x.size(); j++) {
                ex.extract(output_names_x[j], out);
            }
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= loops;
    
    BenchmarkResult benchmarkResult;
    benchmarkResult.loop = loops;
    benchmarkResult.min = time_min;
    benchmarkResult.max = time_max;
    benchmarkResult.avg = time_avg;
    benchmarkResult.width = width;
    benchmarkResult.height = height;

    return benchmarkResult;
};


} // namespace benchmark
