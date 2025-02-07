#include <map>
#include "yolov4.h"

#include "hilog/log.h"

#undef LOG_TAG
#define LOG_TAG "Tncnn"

namespace yolo {


YOLOv4::YOLOv4() {}

YOLOv4::~YOLOv4() { net.clear(); }

int YOLOv4::init(ncnn::Option option, const char *param, const char *model, const char *modeltype) {
    net.opt = option;

    const std::map<std::string, int> _target_sizes = {
        {"yolov4-tiny", 416},
    };

    const std::map<std::string, std::vector<float>> _mean_vals = {
        {"yolov4-tiny", {0.0f, 0.0f, 0.0f}},
    };

    const std::map<std::string, std::vector<float>> _norm_vals = {
        {"yolov4-tiny", {1 / 255.f, 1 / 255.f, 1 / 255.f}},
    };

    target_size = _target_sizes.at(modeltype);
    mean_vals[0] = _mean_vals.at(modeltype)[0];
    mean_vals[1] = _mean_vals.at(modeltype)[1];
    mean_vals[2] = _mean_vals.at(modeltype)[2];
    norm_vals[0] = _norm_vals.at(modeltype)[0];
    norm_vals[1] = _norm_vals.at(modeltype)[1];
    norm_vals[2] = _norm_vals.at(modeltype)[2];

    OH_LOG_DEBUG(LogType::LOG_APP, "load param:%{public}s", param);
    OH_LOG_DEBUG(LogType::LOG_APP, "load bin:%{public}s", model);

    int pr, mr;
//     pr = net.load_param_mem(param); // 加载 content 的
    pr = net.load_param(param);
    mr = net.load_model(model);

    if (pr != 0 || mr != 0) {
        OH_LOG_DEBUG(LogType::LOG_APP, "load param:%{public}d, load model:%{public}d", pr, mr);
    } else {
        OH_LOG_DEBUG(LogType::LOG_APP, "load success");
    }
    return (pr == 0) && (mr == 0);
}

std::vector<BoxInfo> YOLOv4::run(ncnn::Mat &data, int img_w, int img_h, const char *modeltype) {
    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    OH_LOG_DEBUG(LogType::LOG_APP, "letterbox size:%{public}d x %{public}d", w, h);

    ncnn::Mat resize_input = ncnn::Mat::from_pixels_resize(data, ncnn::Mat::PIXEL_RGBA2RGB, img_w, img_h, w, h);
    OH_LOG_DEBUG(LogType::LOG_APP, "mat size:%{public}d x %{public}d x %{public}d", resize_input.w, resize_input.h,
                 resize_input.c);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(resize_input, in_pad, hpad / 2, hpad / 2, wpad / 2, wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    // so for 0-255 input image, rgb_mean should multiply 255 and norm should div by std.
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in_pad);
    ex.extract("output", out);
    auto boxes = decode_infer(out, w + wpad, h + hpad);

    int count = boxes.size();
    OH_LOG_DEBUG(LogType::LOG_APP, "box count:%{public}d", count);
    std::vector<BoxInfo> objects;
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = boxes[i];

        // adjust offset to original unpadded
        float x0 = (objects[i].x1 - (wpad / 2)) / scale;
        float y0 = (objects[i].y1 - (hpad / 2)) / scale;
        float x1 = (objects[i].x2 - (wpad / 2)) / scale;
        float y1 = (objects[i].y2 - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].x1 = x0;
        objects[i].y1 = y0;
        objects[i].x2 = x1;
        objects[i].y2 = y1;
    }

    return objects;
}

std::vector<BoxInfo> YOLOv4::decode_infer(ncnn::Mat &data, int img_w, int img_h) {
    std::vector<BoxInfo> result;
    for (int i = 0; i < data.h; i++) {
        BoxInfo box;
        const float *values = data.row(i);
        box.label = values[0] - 1;
        box.score = values[1];
        box.x1 = values[2] * (float)img_w;
        box.y1 = values[3] * (float)img_h;
        box.x2 = values[4] * (float)img_w;
        box.y2 = values[5] * (float)img_h;
        result.push_back(box);
    }
    return result;
}

} // namespace yolo
