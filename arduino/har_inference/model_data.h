// model_data.h — PLACEHOLDER
//
// This file must be REGENERATED after training and export.
//
// Steps to generate:
//   1. python training/train.py  --data-dir "data/UCI HAR Dataset" --epochs 50
//   2. python training/export.py --data-dir "data/UCI HAR Dataset" \
//                                --checkpoint models/har_cnn_best.pt
//   3. python scripts/tflite_to_c_array.py \
//          --tflite   models/har_cnn_int8.tflite \
//          --output-dir arduino/har_inference/
//
// The generated file will contain the TFLite int8 model as a const byte array.
// Expected size: ~95 KB

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Placeholder declarations — will be populated by tflite_to_c_array.py
extern const unsigned char g_har_model_data[];
extern const unsigned int  g_har_model_len;

#ifdef __cplusplus
}
#endif
