//
// Created by yuan on 2023/1/13.
//

#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <iostream>
#include <mutex>
#include <unordered_map>

#include "bmutility_nn.h"
#include "bmutility_timer.h"
#include "bmutility_pipeline.h"

#include "opencv2/opencv.hpp"

#define CHANNEL_NUM				8
#define FRAME_ROWS				(96)
#define OVERLAP_ROWS			(6)

struct TChannel: public bm::NoCopyable {
    int channel_id;
    uint64_t seq;

    TChannel():channel_id(0), seq(0) {
    }

    ~TChannel() {
        std::cout << "TChannel(chan_id=" << channel_id << ") dtor" <<std::endl;
    }
};

using TChannelPtr = std::shared_ptr<TChannel>;

struct AppStatis {
    int m_channel_num;
    bm::StatToolPtr m_stat_imgps;
    bm::StatToolPtr m_total_fpsPtr;
    uint64_t *m_chan_statis;
    uint64_t m_total_statis = 0;
    std::mutex m_statis_lock;

    AppStatis(int num):m_channel_num(num) {
        m_stat_imgps = bm::StatTool::create(5);
        m_total_fpsPtr = bm::StatTool::create(5);
        m_chan_statis = new uint64_t[m_channel_num];
        assert(m_chan_statis != nullptr);
    }

    ~AppStatis() {
        delete [] m_chan_statis;
    }
};


struct AppSourceFrame {
    int chan_id;
    uint64_t seq;
    // user defined fields
    cv::Mat frame;

    void destroy() {
        frame.release();
    }
};

struct AppForward {
    std::vector<bm_tensor_t> input_tensors;
    std::vector<bm_tensor_t> output_tensors;
};

struct AppFrameInfo {
    std::vector<AppSourceFrame> input_frames;
    std::vector<AppForward> forwards;
    bm_handle_t handle;
    AppFrameInfo(bm_handle_t bmhandle):handle(bmhandle) {}
    void destroy() {
        for (auto& f : input_frames) {
            f.destroy();
        }
        // Free Tensors
        for(auto f : forwards) {
            for (auto &tensor : f.input_tensors) {
                if (tensor.device_mem.size == 0)
                    continue;
                bm_free_device(handle, tensor.device_mem);
                memset(&tensor.device_mem, 0, sizeof(tensor.device_mem));
            }

            for (auto &tensor: f.output_tensors) {
                if (tensor.device_mem.size == 0)
                    continue;
                bm_free_device(handle, tensor.device_mem);
                memset(&tensor.device_mem, 0, sizeof(tensor.device_mem));
            }
        }
    }
};


class App {
    AppStatis &m_appStatis;
    std::shared_ptr<bm::DetectorDelegate<AppFrameInfo>> m_detectorDelegate;
    bm::BMNNContextPtr m_bmctx;
    bm::TimerQueuePtr m_timeQueue;
    int m_channel_start_idx;
    int m_channel_num;
    int m_dev_id;
    int m_max_batch;

    bm::BMInferencePipe<AppFrameInfo> m_inferPipe;
    std::unordered_map<int, TChannelPtr> m_chans;

    uint64_t m_pull_timer_id;
    int m_seq_no;

public:
    App(AppStatis& statis, bm::TimerQueuePtr tq, bm::BMNNContextPtr ctx,
                    int chan_start_index, int chan_num, int max_batch = 1):
            m_detectorDelegate(nullptr), m_channel_num(chan_num), m_bmctx(ctx), m_appStatis(statis)
    {
        m_dev_id = m_bmctx->dev_id();
        m_timeQueue = tq;
        m_channel_start_idx = chan_start_index;
        m_max_batch = max_batch;
        m_pull_timer_id = 0;
    }

    virtual ~App() {
        std::cout << "App dtor, dev_id=" << m_dev_id <<std::endl;
        if (m_pull_timer_id > 0) {
            m_timeQueue->delete_timer(m_pull_timer_id);
            m_pull_timer_id = 0;
        }
    }

    void setDetectorDelegate(std::shared_ptr<bm::DetectorDelegate<AppFrameInfo>> delegate){
        m_detectorDelegate = delegate;
    }

    void run();
};

using AppPtr = std::shared_ptr<App>;

#endif //TEST_MAIN_H
