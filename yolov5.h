//
// Created by yuan on 2023/1/16.
//

#ifndef TEST_YOLOV5_H
#define TEST_YOLOV5_H

#include "base_model.h"

class YoloV5 : public BaseModel {

public:
    YoloV5(bm::BMNNContextPtr ctxPtr);
    ~YoloV5();
protected:
    virtual void postprocess2(AppFrameInfo &frame_info) override;
    virtual void preprocess2(AppFrameInfo &frame_info) override;

};


#endif //TEST_YOLOV5_H
