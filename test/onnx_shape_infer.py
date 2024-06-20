from typing import Union
import argparse

import onnx
import onnx_tool

def onnx_list_find_by_name(onnx_list, name: str):
    for x in onnx_list:
        if x.name == name:
            return x


def change_inputs_batch_size(model: onnx.ModelProto, batch_size: Union[int, None]) -> None:
    """change first dim of all input tensors (convert to dynamic batch_size if 'batch_size' is None)"""
    for t in model.graph.input:
        dim0 = t.type.tensor_type.shape.dim[0]
        if batch_size:
            dim0.dim_value = batch_size
        else:
            dim0.dim_param = 'batch_size'


def shape_inference(model: onnx.ModelProto) -> onnx.ModelProto:

    # # pass: lower Constant Node to initializer
    # i = 0
    # while i < len(model.graph.node):
    #     node = model.graph.node[i]
    #     if node.op_type == 'Constant':
    #         t = onnx_list_find_by_name(node.attribute, "value").t
    #         # print("replace", node.output[0])
    #         # print(t)
    #         tensor = onnx.helper.make_tensor(node.output[0], t.data_type, t.dims, t.raw_data, raw=True)
    #         model.graph.initializer.append(tensor)
    #         model.graph.node.remove(node)
    #     else:
    #         i += 1
    # onnx.checker.check_model(model)

    tmpfile = '/tmp/tmp_model.onnx'
    onnx_tool.model_shape_infer(model, None, saveshapesmodel=tmpfile)
    model = onnx.load(tmpfile)

    # pass: load Constant Node's output shape
    for node in model.graph.node:
        if node.op_type == 'Constant':
            t = onnx_list_find_by_name(node.attribute, "value").t
            value_info = onnx.helper.make_tensor_value_info(node.output[0], t.data_type, t.dims)
            model.graph.value_info.append(value_info)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='')
    parser.add_argument('output', help='')

    args = parser.parse_args()
    model = onnx.load(args.input)
    change_inputs_batch_size(model, 1)
    model = shape_inference(model)
    onnx.save(model, args.output)
