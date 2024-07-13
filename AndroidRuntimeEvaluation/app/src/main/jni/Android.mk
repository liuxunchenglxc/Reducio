LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../native_libs/$(TARGET_ARCH_ABI)/libtensorflowlite.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite_gpu_delegate
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../native_libs/$(TARGET_ARCH_ABI)/libtensorflowlite_gpu_delegate.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := runtimeevaluation
LOCAL_SRC_FILES := com_main_tflmodelruntimeevaluation_RuntimeEvaluation.cpp
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../include
LOCAL_LDFLAGS += $(LOCAL_PATH)/../../native_libs/$(TARGET_ARCH_ABI)/libtensorflowlite.so
LOCAL_LDFLAGS += $(LOCAL_PATH)/../../native_libs/$(TARGET_ARCH_ABI)/libtensorflowlite_gpu_delegate.so
LOCAL_ALLOW_UNDEFINED_SYMBOLS := false
PRODUCT_COPY_FILES += ../libs/$(TARGET_ARCH_ABI)/libtensorflowlite.so:system/lib/libtensorflowlite.so
PRODUCT_COPY_FILES += ../libs/$(TARGET_ARCH_ABI)/libtensorflowlite_gpu_delegate.so:system/lib/libtensorflowlite_gpu_delegate.so
include $(BUILD_SHARED_LIBRARY)

