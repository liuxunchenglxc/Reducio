import tensorflow as tf
import pathlib

def lite_convert(saved_model_dir, tflite_model_save_dir, model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path(tflite_model_save_dir)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    model_filename = model_name + ".tflite"

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_file = tflite_models_dir / model_filename
    tflite_model_file.write_bytes(tflite_model)