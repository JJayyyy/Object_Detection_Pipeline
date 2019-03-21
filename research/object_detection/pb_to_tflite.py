import tensorflow as tf

pb_file = "./ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"
input_arrays = ["Input"]
output_arrays = ["output"]
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(pb_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("frozen_inference_graph.tflite", "wb").write(tflite_model)