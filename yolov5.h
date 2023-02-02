//
// Created by yuan on 2023/1/16.
//

#ifndef TEST_YOLOV5_H
#define TEST_YOLOV5_H

#include "base_model.h"

class YoloV5 : public BaseModel {
//configuration
    float m_confThreshold= 0.5;
    float m_nmsThreshold = 0.5;
    float m_objThreshold = 0.5;
    int argmax(float* data, int dsize);
    float sigmoid(float x);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
    void NMS(std::vector<AppBox> &dets, float nmsConfidence);

public:
    YoloV5(bm::BMNNContextPtr ctxPtr,float confThresh, float objThresh, float nmsThresh);
    ~YoloV5();
protected:
    virtual void postprocess2(AppFrameInfo &frame_info) override;
    virtual void preprocess2(AppFrameInfo &frame_info) override;

};


#endif //TEST_YOLOV5_H
