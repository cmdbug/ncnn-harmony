#ifndef YOLOV4_H
#define YOLOV4_H

#include "net.h"

namespace yolo {

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class YOLOv4 {
public:
    YOLOv4();

    ~YOLOv4();

    int init(ncnn::Option option, const char *param, const char *model, const char *modeltype);
    std::vector<BoxInfo> run(ncnn::Mat &data, int img_w, int img_h, const char *modeltype);

private:
    std::vector<BoxInfo> decode_infer(ncnn::Mat &data, int img_w, int img_h);

    ncnn::Net net;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
};

} // namespace yolo

#endif // YOLOV4_H
