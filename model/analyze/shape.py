from typing import Union
import logging

import onnx
import onnx.helper
import onnx.checker
import onnx_tool

from datatype import TensorsShape
from util import TMPDIR
from .util import onnx_shape_to_list, onnx_list_find_by_name

log = logging.getLogger(__name__)

def get_inputs_shape(model: onnx.ModelProto) -> TensorsShape:
    res = {}
    for inputs in model.graph.input:
        shape = inputs.type.tensor_type.shape
        res[inputs.name] = onnx_shape_to_list(shape)
    return res


def change_inputs_batch_size(model: onnx.ModelProto, batch_size: Union[int, str]) -> None:
    """change first dim of all input tensors (convert to dynamic batch_size if is str)"""
    for t in model.graph.input:
        dim0 = t.type.tensor_type.shape.dim[0]
        if batch_size:
            dim0.dim_value = batch_size
        else:
            dim0.dim_param = batch_size


def shape_inference(model: onnx.ModelProto) -> onnx.ModelProto:

    # pass: lower Constant Node to initializer
    # print("model.graph.node len:", len(model.graph.node))
    # print("model.graph.initializer len:", len(model.graph.initializer))
    i = 0
    while i < len(model.graph.node):
        node = model.graph.node[i]
        if node.op_type == 'Constant':
            t = onnx_list_find_by_name(node.attribute, "value").t
            # print("lower output", node.output[0], "from ConstantOp", node.name, "to initializer")
            # print(t)
            tensor = onnx.helper.make_tensor(node.output[0], t.data_type, t.dims, t.raw_data, raw=True)
            model.graph.initializer.append(tensor)
            model.graph.node.remove(node)
        else:
            i += 1
    # onnx.checker.check_model(model)
    # print("model.graph.node len:", len(model.graph.node))
    # print("model.graph.initializer len:", len(model.graph.initializer))

    tmpfile = str(TMPDIR / "si.onnx")
    onnx_tool.model_shape_infer(model, None, saveshapesmodel=tmpfile, shapesonly=True)
    model_si = onnx.load(tmpfile)

    # copy initializers from origin model, since model_si is shapesonly
    log.debug("model_si.graph.node len: %s" % len(model_si.graph.node))
    # print("model_si.graph.initializer len:", len(model_si.graph.initializer))
    for x in model.graph.initializer:
        model_si.graph.initializer.append(x)
    # print("model_si.graph.initializer len after loaded:", len(model_si.graph.initializer))
    log.debug("model_si.graph.initializer len after loaded: %s" % len(model_si.graph.initializer))
    del model

    # pass: load Constant Node's output shape
    for node in model_si.graph.node:
        if node.op_type == 'Constant':
            # print('Constant', node.name)
            t = onnx_list_find_by_name(node.attribute, "value").t
            value_info = onnx.helper.make_tensor_value_info(node.output[0], t.data_type, t.dims)
            model_si.graph.value_info.append(value_info)

    return model_si
