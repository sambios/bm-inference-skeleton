//
// Created by yuan on 2023/1/15.
//

#ifndef BM_BASE_MODEL_H
#define BM_BASE_MODEL_H

#include "bmutility_pipeline.h"
#include "test_main.h"

class BaseModel : public bm::DetectorDelegate<AppFrameInfo>
{
protected:
    int MAX_BATCH;
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;
    int m_net_w, m_net_h;
public:
    BaseModel(bm::BMNNContextPtr bmctx);
    ~BaseModel();

    virtual int preprocess(std::vector<AppFrameInfo>& frames) override ;
    virtual int forward(std::vector<AppFrameInfo>& frame_info) override ;
    virtual int postprocess(std::vector<AppFrameInfo> &frame_info) override;

protected:
    virtual void postprocess2(AppFrameInfo &frame_info) = 0;
    virtual void preprocess2(AppFrameInfo &frame_info) = 0;
};


#endif //BM_BASE_MODEL_H
