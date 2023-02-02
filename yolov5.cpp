//
// Created by yuan on 2023/1/16.
//

#include "yolov5.h"
#include "bmutility_image.h"

#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

YoloV5::YoloV5(bm::BMNNContextPtr ctxPtr, float confThresh, float objThresh, float nmsThresh):BaseModel(ctxPtr){
    m_confThreshold= confThresh;
    m_objThreshold = objThresh;
    m_nmsThreshold = nmsThresh;
}

YoloV5::~YoloV5() {
    std::cout << "YoloV5 dtor ..." << std::endl;
//    bm_image_free_contiguous_mem(MAX_BATCH, m_resized_imgs.data());
//    bm_image_free_contiguous_mem(MAX_BATCH, m_converto_imgs.data());
//    for(int i=0; i<max_batch; i++){
//        bm_image_destroy(m_converto_imgs[i]);
//        bm_image_destroy(m_resized_imgs[i]);
//    }
}

float YoloV5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w){
        *pIsAligWidth = true;
        ratio = r_w;
    }
    else{
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}
float YoloV5::sigmoid(float x)
{
    return 1.0 / (1 + expf(-x));
}

int YoloV5::argmax(float* data, int num) {
    float max_value = 0.0;
    int max_index = 0;
    for(int i = 0; i < num; ++i) {
        float sigmoid_value = sigmoid(data[i]);
        if (sigmoid_value > max_value) {
            max_value = sigmoid_value;
            max_index = i;
        }
    }

    return max_index;
}

void YoloV5::NMS(std::vector<AppBox> &dets, float nmsConfidence)
{
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const AppBox& a, const AppBox& b) {
        return a.score < b.score;
    });

    std::vector<float> areas(length);
    for (int i=0; i<length; i++)
    {
        areas[i] = dets[i].width * dets[i].height;
    }

    while (index  > 0)
    {
        int i = 0;
        while (i < index)
        {
            float left    = std::max(dets[index].x1,   dets[i].x1);
            float top     = std::max(dets[index].y1,    dets[i].y1);
            float right   = std::min(dets[index].x1 + dets[index].width,  dets[i].x1 + dets[i].width);
            float bottom  = std::min(dets[index].y1 + dets[index].height, dets[i].y1 + dets[i].height);
            float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence)
            {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index --;
            }
            else
            {
                i++;
            }
        }
        index--;
    }
}

void  YoloV5::postprocess2(AppFrameInfo &frame_info) {
    return;
    std::vector<AppBox> yolobox_vec;
    std::vector<cv::Rect> bbox_vec;
    auto& images = frame_info.input_frames;
    for(int batch_idx = 0; batch_idx < (int)images.size(); ++ batch_idx)
    {
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.frame.cols;
        int frame_height = frame.frame.rows;
        int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame_width, frame_height, m_net_w, m_net_h, &isAlignWidth);

        if (isAlignWidth) {
            ty1 = (int)((m_net_h - (int)(frame_height*ratio)) / 2);
        }else{
            tx1 = (int)((m_net_w - (int)(frame_width*ratio)) / 2);
        }
#endif

        int output_num = m_bmnet->outputTensorNum();
        assert(output_num > 0);
        int min_dim = m_bmnet->outputTensor(0)->get_shape()->num_dims;
        int min_idx = 0;
        int box_num = 0;
        for(int i=0; i<output_num; i++){
            auto output_shape = m_bmnet->outputTensor(i)->get_shape();
            auto output_dims = output_shape->num_dims;
            assert(output_dims == 3 || output_dims == 5);
            if(output_dims == 5){
                box_num += output_shape->dims[1] * output_shape->dims[2] * output_shape->dims[3];
            }

            if(min_dim>output_dims){
                min_idx = i;
                min_dim = output_dims;
            }
        }

        auto out_tensor = m_bmnet->outputTensor(min_idx);
        int nout = out_tensor->get_shape()->dims[min_dim-1];
        int class_num = nout - 5;

        float* output_data = nullptr;
        std::vector<float> decoded_data;

        if(min_dim ==3 && output_num !=1){
            std::cout<<"--> WARNING: the current bmodel has redundant outputs"<<std::endl;
            std::cout<<"             you can remove the redundant outputs to improve performance"<< std::endl;
            std::cout<<std::endl;
        }

        if(min_dim == 5){
            std::cout<<"--> Note: Decoding Boxes"<<std::endl;
            std::cout<<"          you can put the process into model during trace"<<std::endl;
            std::cout<<"          which can reduce post process time, but forward time increases 1ms"<<std::endl;
            std::cout<<std::endl;
            const std::vector<std::vector<std::vector<int>>> anchors{
                    {{10, 13}, {16, 30}, {33, 23}},
                    {{30, 61}, {62, 45}, {59, 119}},
                    {{116, 90}, {156, 198}, {373, 326}}};
            const int anchor_num = anchors[0].size();
            assert(output_num == (int)anchors.size());
            assert(box_num>0);
            if((int)decoded_data.size() != box_num*nout){
                decoded_data.resize(box_num*nout);
            }
            float *dst = decoded_data.data();
            for(int tidx = 0; tidx < output_num; ++tidx) {
                auto output_tensor = m_bmnet->outputTensor(tidx);
                int feat_c = output_tensor->get_shape()->dims[1];
                int feat_h = output_tensor->get_shape()->dims[2];
                int feat_w = output_tensor->get_shape()->dims[3];
                int area = feat_h * feat_w;
                assert(feat_c == anchor_num);
                int feature_size = feat_h*feat_w*nout;
                float *tensor_data = (float*)output_tensor->get_cpu_data() + batch_idx*feat_c*area*nout;
                for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++)
                {
                    float *ptr = tensor_data + anchor_idx*feature_size;
                    for (int i = 0; i < area; i++) {
                        dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
                        dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * m_net_h;
                        dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                        dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                        dst[4] = sigmoid(ptr[4]);
                        float score = dst[4];
                        if (score > m_objThreshold) {
                            for(int d=5; d<nout; d++){
                                dst[d] = sigmoid(ptr[d]);
                            }
                        }
                        dst += nout;
                        ptr += nout;
                    }
                }
            }
            output_data = decoded_data.data();
        } else {
            assert(box_num == 0 || box_num == out_tensor->get_shape()->dims[1]);
            box_num = out_tensor->get_shape()->dims[1];
            output_data = (float*)out_tensor->get_cpu_data() + batch_idx*box_num*nout;
        }

        for (int i = 0; i < box_num; i++) {
            float* ptr = output_data+i*nout;
            float score = ptr[4];
            int class_id = argmax(&ptr[5], class_num);
            float confidence = ptr[class_id + 5];
            if (confidence * score > m_objThreshold)
            {
                if (confidence >= m_confThreshold)
                {
                    float centerX = (ptr[0]+1 - tx1)/ratio - 1;
                    float centerY = (ptr[1]+1 - ty1)/ratio - 1;
                    float width = (ptr[2]+0.5) / ratio;
                    float height = (ptr[3]+0.5) / ratio;

                    AppBox box;
                    box.x1 = int(centerX - width / 2);
                    box.y1 = int(centerY - height / 2);
                    box.width = width;
                    box.height = height;
                    box.class_id = class_id;
                    box.score = confidence * score;
                    yolobox_vec.push_back(box);
                }
            }
        }

        printf("\n --> valid boxes number = %d\n", (int)yolobox_vec.size());

#if USE_MULTICLASS_NMS
        std::vector<AppBoxVec> class_vec(class_num);
        for (auto& box : yolobox_vec) {
            class_vec[box.class_id].push_back(box);
        }
        for (auto& cls_box : class_vec){
            NMS(cls_box, m_nmsThreshold);
        }
        yolobox_vec.clear();
        for (auto& cls_box : class_vec){
            yolobox_vec.insert(yolobox_vec.end(), cls_box.begin(), cls_box.end());
        }
#else
        NMS(yolobox_vec, m_nmsThreshold);
#endif
        //LOG_TS(m_ts, "post 3: nms");

        frame_info.input_frames[batch_idx].boxes = yolobox_vec;
    }
}

void  YoloV5::preprocess2(AppFrameInfo &frame_info)
{
    int ret;
    bm_handle_t handle;
    handle = m_bmctx->handle();
    int num = frame_info.input_frames.size();
    //1. Resize
    bm_image resized_images[MAX_BATCH];
    ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, resized_images, num, 64);
    assert(BM_SUCCESS == ret);

    for(int i = 0;i < num; ++i) {
        bm_image image1;
        cv::bmcv::toBMI(frame_info.input_frames[i].frame, &image1, true);
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
            float ratio = get_aspect_scaled_ratio(image1.width, image1.height, m_net_w, m_net_h, &isAlignWidth);
            bmcv_padding_atrr_t padding_attr;
            memset(&padding_attr, 0, sizeof(padding_attr));

            padding_attr.padding_b = 114;
            padding_attr.padding_g = 114;
            padding_attr.padding_r = 114;
            padding_attr.if_memset = 1;
            if (isAlignWidth) {
            padding_attr.dst_crop_h = image1.height*ratio;
            padding_attr.dst_crop_w = m_net_w;

            int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
            padding_attr.dst_crop_sty = ty1;
            padding_attr.dst_crop_stx = 0;
            }else{
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = image1.width*ratio;

            int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = tx1;
            }

            bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
            ret = bmcv_image_vpp_convert_padding(handle, 1, image1, &resized_images[i],
                &padding_attr, &crop_rect);
#else
        ret = bmcv_image_vpp_convert(handle, 1, image1, &resized_images[i]);
        assert(BM_SUCCESS == ret);
#endif
        bm_image_destroy(image1);
    }

    //2. Convert to
    bm_image convertto_imgs[MAX_BATCH];
    float alpha, beta;

    bm_image_data_format_ext img_type = DATA_TYPE_EXT_FLOAT32;
    auto inputTensorPtr = m_bmnet->inputTensor(0);
    if (inputTensorPtr->get_dtype() == BM_INT8) {
        img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
        alpha            = inputTensorPtr->get_scale() * 1.0 / 255;
        beta             = 0.0;
        img_type = (DATA_TYPE_EXT_1N_BYTE_SIGNED);
    }else{
        alpha            = 1.0/255;
        beta             = 0.0;
        img_type = DATA_TYPE_EXT_FLOAT32;
    }

    ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_type, convertto_imgs, num, 1, false, true);
    assert(BM_SUCCESS == ret);

    bm_tensor_t input_tensor = *inputTensorPtr->bm_tensor();
    bm::bm_tensor_reshape_NCHW(handle, &input_tensor, num, 3, m_net_h, m_net_w);

    ret = bm_image_attach_contiguous_mem(num, convertto_imgs, input_tensor.device_mem);
    assert(BM_SUCCESS == ret);

    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = alpha;
    convert_to_attr.alpha_1 = alpha;
    convert_to_attr.alpha_2 = alpha;
    convert_to_attr.beta_0  = beta;
    convert_to_attr.beta_1  = beta;
    convert_to_attr.beta_2  = beta;

    ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr, resized_images, convertto_imgs);
    assert(ret == 0);

    bm_image_dettach_contiguous_mem(num, convertto_imgs);

    AppForward forward1;
    forward1.input_tensors.push_back(input_tensor);
    frame_info.forwards.push_back(forward1);

    bm::BMImage::destroy_batch(resized_images, num);
    bm::BMImage::destroy_batch(convertto_imgs, num);
}