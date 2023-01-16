//
// Created by yuan on 2023/1/15.
//

#include "base_model.h"
#include "bmutility_image.h"

BaseModel::BaseModel(bm::BMNNContextPtr bmctx):m_bmctx(bmctx)
{
    // the bmodel has only one yolo network.
    auto net_name = m_bmctx->network_name(0);
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), net_name);
    assert(m_bmnet != nullptr);
    assert(m_bmnet->inputTensorNum() == 1);

    //BaseModel input is NCHW
    auto tensor = m_bmnet->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    // read batch from model
    MAX_BATCH = tensor->get_shape()->dims[0];
}

BaseModel::~BaseModel()
{

}

int BaseModel::preprocess(std::vector<AppFrameInfo>& input_frames)
{
    int ret = 0;

    bm_handle_t handle = m_bmctx->handle();
    //total calculate frame number
    std::vector<AppSourceFrame> vecFrameBaseInfo;
    for(int frameInfoIdx = 0; frameInfoIdx < input_frames.size(); ++frameInfoIdx) {
        auto &frame_info = input_frames[frameInfoIdx];
        vecFrameBaseInfo.insert(vecFrameBaseInfo.end(), frame_info.input_frames.begin(), frame_info.input_frames.end());
    }

    //Clear the input frames, because we'll re-arrange it later.
    input_frames.clear();
    int total = vecFrameBaseInfo.size();
    int left = (total%MAX_BATCH == 0 ? MAX_BATCH: total%MAX_BATCH);
    int batch_num = total%MAX_BATCH==0 ? total/MAX_BATCH: (total/MAX_BATCH + 1);

    for(int batch_idx = 0; batch_idx < batch_num; ++ batch_idx) {
        int num = MAX_BATCH;
        int start_idx = batch_idx*MAX_BATCH;
        if (batch_idx == batch_num-1) {
            // last one
            num = left;
        }

        AppFrameInfo new_frame_info(handle);
        for(int i = 0; i < num; ++i) {
            new_frame_info.input_frames.push_back(vecFrameBaseInfo[start_idx+i]);
        }

        // do real preprocess
        preprocess2(new_frame_info);

        input_frames.push_back(new_frame_info);
    }

    return ret;
}

int BaseModel::forward(std::vector<AppFrameInfo>& frame_infos)
{
    int ret = 0;
    for(int b = 0; b < frame_infos.size(); ++b) {
        for(int l = 0; l < frame_infos[b].forwards.size(); ++l) {
            ret = m_bmnet->forward(frame_infos[b].forwards[l].input_tensors.data(),
                                  frame_infos[b].forwards[l].input_tensors.size(),
                                  frame_infos[b].forwards[l].output_tensors.data(),
                                  frame_infos[b].forwards[l].output_tensors.size());
            assert(BM_SUCCESS == ret);
        }
    }
    return ret;
}

int BaseModel::postprocess(std::vector<AppFrameInfo> &frame_infos)
{
    int ret = 0;
    for(int frameInfoIdx =0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {

        // Free AVFrames
        auto &frame_info = frame_infos[frameInfoIdx];

        // extract face detection
        postprocess2(frame_info);

        // Free Tensors
        for (auto &f : frame_info.forwards) {
            for (auto &tensor : f.input_tensors) {
                bm_free_device(m_bmctx->handle(), tensor.device_mem);
            }

            for (auto &tensor: f.output_tensors) {
                bm_free_device(m_bmctx->handle(), tensor.device_mem);
            }
        }

        frame_info.forwards.clear();

        if (m_nextInferPipe == nullptr) {
            // Last pipeline, callback the result
            if (m_pfnDetectFinish != nullptr) {
                m_pfnDetectFinish(frame_info);
            }

        } else {
            // transfer to next pipe
            m_nextInferPipe->push_frame(&frame_info);
        }

    }
    return ret;
}
