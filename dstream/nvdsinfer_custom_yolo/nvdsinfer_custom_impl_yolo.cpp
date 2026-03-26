#include <iostream>
#include <vector>
#include <string>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    if (outputLayersInfo.empty()) return false;

    // Safety: Grab the FIRST layer (usually the only one for YOLOv8)
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    
    float* output = (float*)layer.buffer;
    if (!output) return false;

    // YOLOv8 shape is usually [1, 84, 8400]
    // d[0] = 84 (4 coords + 80 classes)
    // d[1] = 8400 (boxes)
    unsigned int num_rows = layer.inferDims.d[0]; 
    unsigned int num_boxes = layer.inferDims.d[1];
    unsigned int num_classes = detectionParams.numClassesConfigured;

    // DEBUG: Only print once to avoid log spam
    static bool first_run = true;
    if (first_run) {
        std::cout << "CHECK: LayerName=" << layer.layerName 
                  << " Rows=" << num_rows 
                  << " Boxes=" << num_boxes << std::endl;
        first_run = false;
    }

    for (unsigned int i = 0; i < num_boxes; i++) {
        float max_score = 0;
        int max_class = -1;

        for (unsigned int c = 0; c < num_classes; c++) {
            // CRITICAL: Calculate index carefully
            unsigned int index = (4 + c) * num_boxes + i;
            
            // This is the most likely crash point if num_boxes is wrong
            float score = output[index]; 
            
            if (score > 0.25f) {
                if (score > max_score) {
                    max_score = score;
                    max_class = c;
                }
            }
        }

        if (max_class != -1) {
            NvDsInferObjectDetectionInfo obj;
            // Center X, Center Y, Width, Height
            float xc = output[0 * num_boxes + i];
            float yc = output[1 * num_boxes + i];
            float w  = output[2 * num_boxes + i];
            float h  = output[3 * num_boxes + i];

            obj.left = (xc - w / 2.0f);
            obj.top = (yc - h / 2.0f);
            obj.width = w;
            obj.height = h;
            obj.detectionConfidence = max_score;
            obj.classId = max_class;
            objectList.push_back(obj);
        }
    }
    return true;
}



/* bash

$ g++ -o libnvdsinfer_custom_impl_yolo.so -shared -fPIC nvdsinfer_custom_impl_yolo.cpp \
-I /opt/nvidia/deepstream/deepstream-7.1/sources/includes/ \
-I /usr/local/cuda/include/ \
-L /opt/nvidia/deepstream/deepstream-7.1/lib/ \
-L /usr/local/cuda/lib64/ \
-L /usr/lib/aarch64-linux-gnu/ \
-lnvds_infer -lcudart


*/
