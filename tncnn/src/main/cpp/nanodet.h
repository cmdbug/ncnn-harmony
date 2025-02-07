#ifndef NANODET_H
#define NANODET_H

#include "net.h"
#include <string>

namespace nanodet {

typedef struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class NanoDet {
public:
    NanoDet();

    ~NanoDet();

    
    int init(ncnn::Option option, const char *param, const char *model, const char *modeltype);
    std::vector<BoxInfo> run(ncnn::Mat &data, int img_w, int img_h, const char *modeltype);

private:
    void decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride, float threshold,
                      std::vector<std::vector<BoxInfo>> &results, float width_ratio, float height_ratio);

    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride, float width_ratio,
                         float height_ratio);

    static void nms(std::vector<BoxInfo> &result, float nms_threshold);

    ncnn::Net net;
    int target_size = 320;
    float mean_vals[3];
    float norm_vals[3];
    int num_class = 80;
    int reg_max = 7;
    std::vector<HeadInfo> heads_info{
        // cls_pred|dis_pred|stride
        {"792", "795", 8},
        {"814", "817", 16},
        {"836", "839", 32},
    };
};

} // namespace nanodet

#endif // NANODET_H
