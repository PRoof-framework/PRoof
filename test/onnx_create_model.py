import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto


N = 1024
REPEAT = 4

input = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch_size', N, N])

nodes = []
next_input = 'input'
for i in range(REPEAT):
    inner = f'out{i}'
    nodes.append(
        helper.make_node(
            "MatMul",                   # type
            [next_input, 'input'],      # inputs
            [inner],                    # outputs
            f"malmul_{i}"               # name
        )
    )
    next_input = inner

output = helper.make_tensor_value_info(next_input, TensorProto.FLOAT, ['batch_size', N, N])

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    nodes,              # nodes
    "test-model",       # name
    [input],            # inputs
    [output],           # outputs
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name="proof")

print(f"The model is:\n{model_def}")
onnx.checker.check_model(model_def)
print("The model is checked!")
onnx.save(model_def, 'tmp/gemm.onnx')
