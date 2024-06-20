import json
from PIL import Image
import numpy as np

import onnx
import onnxruntime as ort

def change_inputs_batch_size(model: onnx.ModelProto, batch_size: int) -> None:
    for inputs in model.graph.input:
        dim0 = inputs.type.tensor_type.shape.dim[0]
        dim0.dim_value = batch_size

BATCH_SIZE = 128
MODEL = 'model/resnet50.onnx'
# MODEL = 'model/ckpt_kadid10k.pt-opset16.onnx'

model = onnx.load(MODEL)

change_inputs_batch_size(model, BATCH_SIZE)

image_path = "data/drum.jpg"
img = Image.open(image_path).resize((256, 256)).crop((16, 16, 240, 240))
input_data = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
input_data -= np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
input_data /= np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
input_data = input_data.reshape((1, 3, 224, 224)).repeat(BATCH_SIZE, axis=0)

x = input_data
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0
# sess_options.log_verbosity_level = 0
sess_options.enable_profiling = True
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# sess_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, 'ORT_ENABLE_EXTENDED')
# sess_options.profile_file_prefix = '/tmp/ort_profile'

sess_options.optimized_model_filepath = "model/opti.onnx"
ort_sess = ort.InferenceSession(model.SerializeToString(), sess_options)

outputs = ort_sess.run(None, {'input': x})
outputs = ort_sess.run(None, {'input': x})
outputs = ort_sess.run(None, {'input': x})

prof_file = ort_sess.end_profiling()

with open(prof_file) as f:
    prof = json.load(f)
    for i in prof:
        if i['name'] == 'model_run':
            print('model_run', i['dur'])


# Print Result
with open('data/imagenet_classes.txt') as f:
    classes = [x.strip() for x in f.readlines()]
predicted = classes[outputs[0][0].argmax(0)]
print(f'Predicted: "{predicted}"')

# import onnx
# model = onnx.load('model/resnet50.onnx')
# print(len(model.graph.initializer))
# print(len(model.graph.node))

