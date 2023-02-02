//
// Created by yuan on 2023/1/13.
//
#include "opencv2/opencv.hpp"

#include "test_main.h"
#include "yolov5.h"


void App::run() {
    m_seq_no = 0;
    m_detectorDelegate->set_detected_callback([this](AppFrameInfo &frameInfo) {
        std::cout << ">>>>" << std::endl;
        uint64_t current_ts = bm::gettime_msec();
        for (int i = 0; i < frameInfo.input_frames.size(); ++i) {
            int ch = frameInfo.input_frames[i].chan_id;
            auto delta = current_ts - frameInfo.input_frames[i].ts;
            printf("[ch=%d, seq=%lu]t1=%lu, t2=%lu, latency=%lu ms\n", ch, frameInfo.input_frames[i].seq,
                    current_ts, frameInfo.input_frames[i].ts, delta);
            m_appStatis.m_chan_statis[ch]++;
            m_appStatis.m_total_statis++;

        }
        std::cout << "<<<<" << std::endl;
    });

    bm::DetectorParam param;
    int cpu_num = std::thread::hardware_concurrency();
    int tpu_num = 1;

    param.preprocess_thread_num = cpu_num;
    param.preprocess_queue_size = std::max(m_channel_num, 8);
    param.inference_thread_num = tpu_num;
    param.inference_queue_size = m_channel_num;
    param.postprocess_thread_num = cpu_num;
    param.postprocess_queue_size = m_channel_num;
    param.batch_num = m_max_batch;

    m_inferPipe.init(param, m_detectorDelegate);

    //read image from path
    cv::Mat m = cv::imread("zidane.jpg");
    if (m.empty()) {
        printf("read zidane.jpg failed!");
        exit(0);
    }
    m_timeQueue->create_timer(60, [this, m]{
        AppSourceFrame frameInfo;
        AppFrameInfo input_frame(m_bmctx->handle());
        // fill frame
        auto ts = bm::gettime_msec();
        for(int ch = 0; ch < CHANNEL_NUM; ++ch) {
            frameInfo.chan_id = ch;
            frameInfo.seq = m_seq_no++;
            frameInfo.frame = m;
            frameInfo.ts = ts;
            input_frame.input_frames.push_back(frameInfo);
            m_inferPipe.push_frame(&input_frame);
        }

    }, 1, &m_pull_timer_id);
}


int main(int argc, char *argv[])
{
    const char *base_keys=
            "{help | 0 | Print help information.}"
            "{bmodel | compilation.bmodel | Bmodel file path}"
            "{max_batch | 4 | max batch num}";

    std::string keys;
    keys = base_keys;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    AppStatis appStatis(CHANNEL_NUM);
    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
    int max_batch = parser.get<int>("max_batch");
    std::string bmodel_path = parser.get<std::string>("bmodel");
    int dev_id = 0;

    // init
    bm::BMNNHandlePtr handle = std::make_shared<bm::BMNNHandle>(dev_id);
    bm::BMNNContextPtr contextPtr = std::make_shared<bm::BMNNContext>(handle, bmodel_path);
    //bmlib_log_set_level(BMLIB_LOG_VERBOSE);

    // create app
    AppPtr appPtr = std::make_shared<App>(appStatis, tqp, contextPtr,  0, CHANNEL_NUM, 4);

    // set detector delegator
    std::shared_ptr<YoloV5> detector = std::make_shared<YoloV5>(contextPtr, 0.5, 0.5, 0.5);
    appPtr->setDetectorDelegate(detector);
    appPtr->run();

    uint64_t timer_id;
    tqp->create_timer(1000, [&appStatis](){
        int ch = 0;
        appStatis.m_stat_imgps->update(appStatis.m_chan_statis[ch]);
        appStatis.m_total_fpsPtr->update(appStatis.m_total_statis);

        double imgfps = appStatis.m_stat_imgps->getSpeed();
        double totalfps = appStatis.m_total_fpsPtr->getSpeed();

        std::cout << "[" << bm::timeToString(time(0)) << "] total fps ="
                  << std::setiosflags(std::ios::fixed) << std::setprecision(1) << totalfps
                  <<  ",ch=" << ch << ": speed=" << imgfps << std::endl;
    }, 1, &timer_id);

    tqp->run_loop();

    return 0;
}