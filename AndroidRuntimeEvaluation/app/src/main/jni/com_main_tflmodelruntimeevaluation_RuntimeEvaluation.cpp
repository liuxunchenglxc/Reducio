#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/delegates/serialization.h"
#include "com_main_tflmodelruntimeevaluation_RuntimeEvaluation.h"
#include <ctime>
#include "cpuinfo/cpuinfo.h"

int64_t NowMicros() {
    struct timeval tv{};
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}


extern "C" JNIEXPORT jdouble JNICALL Java_com_main_tflmodelruntimeevaluation_RuntimeEvaluation_runtimeEvaluate
    (JNIEnv *env, jclass clazz, jstring path, jstring cachePath) {
    const char* filename = env->GetStringUTFChars(path, nullptr);
    const char* cachename = env->GetStringUTFChars(cachePath, nullptr);
    std::unique_ptr<tflite::FlatBufferModel> model=tflite::FlatBufferModel::BuildFromFile(filename);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.is_precision_loss_allowed = 1;
    options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
    options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    TfLiteDelegate* delegate = TfLiteGpuDelegateV2Create(&options);
    interpreter->ModifyGraphWithDelegate(delegate);
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->AllocateTensors();
    auto* input = interpreter->typed_input_tensor<float >(0);
    *input=3.3;
    interpreter->Invoke();
    interpreter->Invoke();
    interpreter->Invoke();
    int64_t start_time, end_time;
    int mi = 10;
    int mai = 100;
    int64_t mt = 5000000;
    int64_t total_duration = 0;
    int iter = 0;
    for (int i = 0; i < mi || (total_duration < mt && i < mai);
         ++i) {
        start_time = NowMicros();
        interpreter->Invoke();
        end_time = NowMicros();
        int64_t duration = end_time - start_time;
        total_duration += duration;
        ++iter;
    }
    double run_ms = total_duration * 1e-3 / iter;
    TfLiteGpuDelegateV2Delete(delegate);
    env->ReleaseStringUTFChars(path, filename);
    return run_ms;
}
