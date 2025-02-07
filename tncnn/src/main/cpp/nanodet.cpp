#include "nanodet.h"
#include <map>

#include "hilog/log.h"

#undef LOG_TAG
#define LOG_TAG "Tncnn"

namespace nanodet {

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

template <typename _Tp> int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

NanoDet::NanoDet() {}

NanoDet::~NanoDet() { net.clear(); }

int NanoDet::init(ncnn::Option option, const char *param, const char *model, const char *modeltype) {
    net.opt = option;

    const std::map<std::string, int> _target_sizes = {
        {"nanodet-m", 320},
    };

    const std::map<std::string, std::vector<float>> _mean_vals = {
        {"nanodet-m", {103.53f, 116.28f, 123.675f}},
    };

    const std::map<std::string, std::vector<float>> _norm_vals = {
        {"nanodet-m", {0.017429f, 0.017507f, 0.01712475}},
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


std::vector<BoxInfo> NanoDet::run(ncnn::Mat &data, int img_w, int img_h, const char *modeltype) {
    float width_ratio = (float)img_w / (float)target_size;
    float height_ratio = (float)img_h / (float)target_size;

    ncnn::Mat resize_input =
        ncnn::Mat::from_pixels_resize(data, ncnn::Mat::PIXEL_RGBA2BGR, img_w, img_h, target_size, target_size);

    resize_input.substract_mean_normalize(mean_vals, norm_vals);
    
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", resize_input);
    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class);

    for (const auto &head_info : this->heads_info) {
        ncnn::Mat dis_pred;
        ncnn::Mat cls_pred;
        ex.extract(head_info.dis_layer.c_str(), dis_pred);
        ex.extract(head_info.cls_layer.c_str(), cls_pred);

        decode_infer(cls_pred, dis_pred, head_info.stride, 0.3f, results, width_ratio, height_ratio);
    }

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++) {
        nms(results[i], 0.7f);

        for (auto box : results[i]) {
            dets.push_back(box);
        }
    }
    return dets;
}


void NanoDet::decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride, float threshold,
                           std::vector<std::vector<BoxInfo>> &results, float width_ratio, float height_ratio) {
    int feature_h = target_size / stride;
    int feature_w = target_size / stride;

    // cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
        const float *scores = cls_pred.row(idx);
        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            // std::cout << "label:" << cur_label << " score:" << score << std::endl;
            const float *bbox_pred = dis_pred.row(idx);
            results[cur_label].push_back(
                this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride, width_ratio, height_ratio));
            // debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
            // cv::imshow("debug", debug_heatmap);
        }
    }
}

BoxInfo NanoDet::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride,
                              float width_ratio, float height_ratio) {
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float *dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        // std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f) * width_ratio;
    float ymin = (std::max)(ct_y - dis_pred[1], .0f) * height_ratio;
    float xmax = (std::min)(ct_x + dis_pred[2], (float)target_size) * width_ratio;
    float ymax = (std::min)(ct_y + dis_pred[3], (float)target_size) * height_ratio;

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void NanoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] =
            (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

} // namespace nanodet
