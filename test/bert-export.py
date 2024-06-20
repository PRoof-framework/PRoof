import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# load model and tokenizer
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dummy_model_input = tokenizer("something bad", return_tensors="pt")

with torch.no_grad():
    logits = model(**dummy_model_input).logits

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])

seq = 'sequence'

# export
torch.onnx.export(
    model,
    tuple(dummy_model_input.values()),
    f="torch-model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: seq},
                  'attention_mask': {0: 'batch_size', 1: seq},
                  'logits': {0: 'batch_size', 1: seq}},
    do_constant_folding=True,
    opset_version=13,
)

import onnx
import onnxruntime as ort

def change_inputs_batch_size(model: onnx.ModelProto, batch_size: int) -> None:
    for inputs in model.graph.input:
        dim0 = inputs.type.tensor_type.shape.dim[0]
        dim0.dim_value = batch_size

MODEL = 'torch-model.onnx'

model = onnx.load(MODEL)


sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0
# sess_options.log_verbosity_level = 0
# sess_options.enable_profiling = True
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# sess_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, 'ORT_ENABLE_EXTENDED')
# sess_options.profile_file_prefix = '/tmp/ort_profile'

# sess_options.optimized_model_filepath = "model/opti.onnx"
ort_sess = ort.InferenceSession(model.SerializeToString(), sess_options)

print(dummy_model_input)
outputs = ort_sess.run(None, {k: v.numpy() for k, v in dummy_model_input.items()})

prof_file = ort_sess.end_profiling()

predicted = outputs
print(f'Predicted: "{predicted}"')

