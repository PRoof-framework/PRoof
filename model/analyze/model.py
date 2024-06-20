from __future__ import annotations
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging
import itertools
import onnx

from datatype import AnalyzeData, TensorShape, NodeData
from .util import onnx_shape_to_list
from .op import OpRegister, UnknownOp, _BaseOp
from .shape import change_inputs_batch_size, shape_inference, get_inputs_shape

log = logging.getLogger(__name__)


def onnx_model_layer_profile(model: onnx.ModelProto) -> Tuple[AnalyzeData, Dict[str, _BaseOp]]:
    """
    profile the model layer, assume batch_size=1 if the model has dynamic batch_size

    return format:

    - [dict]
        - nodes: dict
            - <name>: dict
                - name: str
                - type: str
                - flops: int
                - memory: int
                - params: int
                - inputs: list
                    - [dict]
                        - name: str
                        - shape: list
                - outputs: list
                    - [dict]
                        - name: str
                        - shape: list
        - total_flops: int
        - total_memory: int
        - total_params: int

    """

    data = AnalyzeData()
    data.inputs = get_inputs_shape(model)
    log.debug("model inputs: %s", data.inputs)

    change_inputs_batch_size(model, 1)
    model = shape_inference(model)

    nodes_list = [n.name for n in model.graph.node]
    nodes = {n.name: n for n in model.graph.node}

    initializer_dict = {
        i.name: {
            'name': i.name,
            'dims': list(i.dims),
            'data_type': i.data_type,
        } for i in model.graph.initializer
    }

    shape_dict = {
        v.name: {
            'name': v.name,
            'dims': onnx_shape_to_list(v.type.tensor_type.shape),
        } for v in itertools.chain(
            model.graph.input,
            model.graph.output,
            model.graph.value_info,
        )
    }

    data.nodes = {}
    op_dict = {}
    for name in nodes_list:
        node = nodes[name]

        op_type = node.op_type
        inputs: List[TensorShape] = []
        params_list: List[str] = []
        for input_name in node.input:
            shape = []
            if input_name in shape_dict:
                shape = shape_dict[input_name]['dims']
            if input_name in initializer_dict:
                if shape:   # if also in shape_dict
                    assert shape == initializer_dict[input_name]['dims']
                shape = initializer_dict[input_name]['dims']
                params_list.append(input_name)

            # if not shape:
            #     log.debug("shape of tensor %s (input of Node %s) not found, assume it is scalar (as design if it is)", input_name, name)

            inputs.append(TensorShape(
                name = input_name,
                shape = shape
            ))

        outputs: List[TensorShape] = []
        for output_name in node.output:
            if output_name in shape_dict:
                shape = shape_dict[output_name]['dims']
            elif output_name in initializer_dict:
                shape = initializer_dict[output_name]['dims']
            else:
                shape = []
                # log.warning("shape of tensor %s (output of Node %s) not found, assume it is scalar (as design if it is)", output_name, name)

            outputs.append(TensorShape(
                name = output_name,
                shape = shape
            ))

        # _BaseOp(name: str, type: str, inputs: List[dict], outputs: List[dict], params_list: List[str], onnx_node: onnx.NodeProto, flops_table: dict = DEFAULT_FLOPS_TABLE)
        op_class = OpRegister.get(op_type)
        op: _BaseOp = op_class(name, op_type, inputs, outputs, params_list, node)
        op_dict[name] = op
        if type(op) == UnknownOp:
            log.debug("unsupport op_type: %s, fallback to UnknownOp", op_type)

        node_data = NodeData()
        node_data.name = name
        node_data.type = op_type
        node_data.flops = op.get_flops()
        node_data.memory = op.get_memory()
        node_data.params = op.get_params()
        node_data.inputs = inputs
        # node_data.inputs = [x for x in inputs if x.name not in params_list]
        node_data.outputs = outputs

        data.nodes[name] = node_data

    data.total_flops = sum(n.flops for n in data.nodes.values())
    data.total_memory = sum(n.memory for n in data.nodes.values())
    data.total_params = sum(n.params for n in data.nodes.values())
    log.info("the model has %s nodes", len(data.nodes))
    log.info("total FLOPs %.3f M", data.total_flops / 1e6)
    log.info("total memory %.3f M (vars) (approximate, not fused)", data.total_memory / 1e6)
    log.info("total params %.3f M (vars)", data.total_params / 1e6)

    return data, op_dict
