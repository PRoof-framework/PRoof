import onnx

def onnx_shape_to_list(shape: onnx.TensorShapeProto) -> list:
    return [d.dim_param or int(d.dim_value) for d in shape.dim]

def onnx_list_find_by_name(onnx_list, name: str):
    for x in onnx_list:
        if x.name == name:
            return x
