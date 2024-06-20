from __future__ import annotations
from typing import List, Set
from math import prod

import onnx

from datatype import TensorShape
from .flops import DEFAULT_FLOPS_TABLE
from .util import onnx_list_find_by_name


def _get_attribute_value(onnx_node: onnx.NodeProto, attribute: str):
    return tuple(map(int, onnx_list_find_by_name(onnx_node.attribute, attribute).ints))


def _get_axes(shape: list, axes: list) -> list:
    return [shape[i] for i in axes]


class _BaseOp():
    "note: each _BaseOp should has a unique name in a network"
    def __init__(self, name: str, type: str, inputs: List[TensorShape], outputs: List[TensorShape], params_list: List[str], onnx_node: onnx.NodeProto, flops_table: dict = DEFAULT_FLOPS_TABLE):
        self.name: str = name
        self.type: str = type
        self.inputs: List[TensorShape] = inputs    # also include params
        self.outputs: List[TensorShape] = outputs
        self.params: Set[str] = set(params_list)
        self._onnx_node = onnx_node
        self._flops = flops_table
        self._extra_info(onnx_node)

    def _extra_info(self, onnx_node: onnx.NodeProto) -> None:
        pass

    def __repr__(self) -> str:
        # return f"{self.__class__.__name__}.{self.name}(inputs={[x.name for x in self.inputs]} --> outputs={[x.name for x in self.outputs]})"
        return f"{self.__class__.__name__}({self.name})"

    def as_op(self, op_class: _BaseOp):
        return op_class(self.name, self.type, self.inputs, self.outputs, self.params, self._onnx_node, self._flops)

    # default
    def get_input_size(self, input_idx: int) -> int:
        return self.inputs[input_idx].size()

    def get_output_size(self, output_idx: int) -> int:
        return self.outputs[output_idx].size()

    def get_flops(self) -> int:
        """return total FLOPs of this op (not MACs, and 1MACs = 2FLOPs)"""
        return 0

    def get_memory(self, batch_size=1) -> int:
        """return approximate total memory access count of variables, include params, e.g. 4vars == 16Bytes when datatype is fp32"""
        s = 0
        for i, v in enumerate(self.inputs):
            if v.name in self.params:
                s += self.get_input_size(i)
            else:
                s += self.get_input_size(i) * batch_size
        for i, v in enumerate(self.outputs):
            s += self.get_output_size(i) * batch_size
        return s

    def get_params(self) -> int:
        """return total params (variables) count"""
        return sum(prod(x.shape) for x in self.inputs
            if x.name in self.params)


class OpRegister():
    op_dict = {}

    @classmethod
    def register(cls, op_class: _BaseOp) -> _BaseOp:
        assert op_class.__name__.endswith('Op')
        op_name = op_class.__name__[:-len('Op')]
        cls.op_dict[op_name] = op_class
        return op_class

    @classmethod
    def get(cls, name: str) -> _BaseOp:
        return cls.op_dict.get(name) or UnknownOp


class UnknownOp(_BaseOp):
    pass


@OpRegister.register
class ConcatOp(_BaseOp):
    pass


@OpRegister.register
class SplitOp(_BaseOp):
    pass


@OpRegister.register
class SliceOp(_BaseOp):
    def get_input_size(self, input_idx: int) -> int:
        if input_idx == 0:
            # not all data of input is need to read
            return self.get_output_size(0)
        return super().get_input_size(input_idx)


@OpRegister.register
class ExpandOp(_BaseOp):
    pass


@OpRegister.register
class TransposeOp(_BaseOp):
    pass


@OpRegister.register
class GatherOp(_BaseOp):
    pass


### Dummy ###

class _DummyOp(_BaseOp):
    def get_memory(self, batch_size=1) -> int:
        # only when standalone. Or _FusedOp will override this if such a _DummyOp has been fused
        return 0


@OpRegister.register
class FlattenOp(_DummyOp):
    pass


@OpRegister.register
class ConstantOp(_DummyOp):
    pass


@OpRegister.register
class IdentityOp(_DummyOp):
    pass


@OpRegister.register
class ShapeOp(_DummyOp):
    def get_input_size(self, input_idx: int) -> int:
        return 0


@OpRegister.register
class SqueezeOp(_DummyOp):
    pass


@OpRegister.register
class UnsqueezeOp(_DummyOp):
    pass


@OpRegister.register
class ReshapeOp(_DummyOp):
    pass


### Pointwise ###

class _PointwiseOp(_BaseOp):
    def flops_per_element(self) -> int:
        """FLOPs per element in output tensors"""
        return self._flops['_UNKNOWN']

    def get_flops(self) -> int:
        output_elements = prod(self.outputs[0].shape)
        return output_elements * self.flops_per_element()


@OpRegister.register
class LogOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['LOG']


@OpRegister.register
class PowOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['POW']


@OpRegister.register
class SqrtOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['SQRT']


@OpRegister.register
class ErfOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['ERF']


@OpRegister.register
class SigmoidOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['ADD'] + self._flops['LOG'] + self._flops['ADD'] + self._flops['DIV']


@OpRegister.register
class SoftmaxOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['ADD'] + self._flops['EXP'] + self._flops['DIV']


@OpRegister.register
class ReluOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['CMP']


@OpRegister.register
class ClipOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['CMP'] * 2


@OpRegister.register
class AddOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['ADD']


@OpRegister.register
class SubOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['ADD']


@OpRegister.register
class MulOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['MUL']


@OpRegister.register
class DivOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['DIV']


@OpRegister.register
class CastOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['CAST']


@OpRegister.register
class WhereOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['CMP']


@OpRegister.register
class EqualOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['CMP']


@OpRegister.register
class ConstantOfShapeOp(_PointwiseOp):
    def flops_per_element(self) -> int:
        return self._flops['CMP']


### Reduction ###

class _ReductionOp(_BaseOp):
    def flops_per_compute_and_reduce(self) -> int:
        """FLOPs per compute and reduce, like multiply-accumulate in Gemm"""
        return self._flops['_UNKNOWN']

    def flops_per_element_after_reduce(self) -> int:
        """FLOPs per compute and reduce, like add bias in Gemm"""
        return 0

    def reduce_count(self) -> int:
        """count of values which reduced to an element in output tensors"""
        return self.inputs[-1].shape[-1]

    def get_flops(self) -> int:
        output_elements = prod(self.outputs[0].shape)
        return (output_elements * self.reduce_count() * self.flops_per_compute_and_reduce()
            + output_elements * self.flops_per_element_after_reduce())


@OpRegister.register
class MatMulOp(_ReductionOp):
    def flops_per_compute_and_reduce(self) -> int:
        return self._flops['MAC']

    def reduce_count(self) -> int:
        a_shape = self.inputs[0].shape
        return a_shape[-1]


@OpRegister.register
class GemmOp(_ReductionOp):
    def flops_per_compute_and_reduce(self) -> int:
        return self._flops['MAC']

    def flops_per_element_after_reduce(self) -> int:
        if len(self.inputs) == 3:   # bias is optional
            return self._flops['ADD']
        else:
            return 0

    def reduce_count(self) -> int:
        x_shape = self.inputs[0].shape
        # FIXME: when transA is 1, this is not correct
        return x_shape[1]  # M, K


@OpRegister.register
class ConvOp(GemmOp):
    def _extra_info(self, onnx_node: onnx.NodeProto) -> None:
        self.kernel_shape = _get_attribute_value(onnx_node, 'kernel_shape')
        self.strides = _get_attribute_value(onnx_node, 'strides')

    def reduce_count(self) -> int:
        # img2col like
        group_kernel_shape = self.inputs[1].shape[1:]  # N, C, K1, K2 ... KN
        return prod(group_kernel_shape)

    def get_input_size(self, input_idx: int) -> int:
        if input_idx == 0:
            # not all elements in inputs will used, calculate kernel occupy on stride step square
            # TODO: dilation conv not supported yet
            trimmed_kernel_shape = tuple(map(min, zip(self.kernel_shape, self.strides)))
            occupy = prod(trimmed_kernel_shape) / prod(self.strides)
            return int(super().get_input_size(input_idx) * occupy)
        return super().get_input_size(input_idx)


@OpRegister.register
class GlobalAveragePoolOp(_ReductionOp):
    def flops_per_compute_and_reduce(self) -> int:
        return self._flops["ADD"]

    def flops_per_element_after_reduce(self) -> int:
        return self._flops["MUL"]   # may use MUL instead of DIV

    def reduce_count(self) -> int:
        x_shape = self.inputs[0].shape
        return prod(x_shape[2:])   # N, C, D1, D2 ... DN


@OpRegister.register
class MaxPoolOp(_ReductionOp):
    def _extra_info(self, onnx_node: onnx.NodeProto) -> None:
        self.kernel_shape = _get_attribute_value(onnx_node, 'kernel_shape')

    def flops_per_compute_and_reduce(self) -> int:
        return self._flops['CMP']

    def reduce_count(self) -> int:
        return prod(self.kernel_shape)


@OpRegister.register
class ReduceMeanOp(_ReductionOp):
    def _extra_info(self, onnx_node: onnx.NodeProto) -> None:
        self.axes = _get_attribute_value(onnx_node, 'axes')

    def flops_per_compute_and_reduce(self) -> int:
        return self._flops['ADD']

    def flops_per_element_after_reduce(self) -> int:
        return self._flops["MUL"]   # may use MUL instead of DIV

    def reduce_count(self) -> int:
        return prod(_get_axes(self.inputs[0].shape, self.axes))


@OpRegister.register
class ReduceSumOp(_ReductionOp):
    def flops_per_compute_and_reduce(self) -> int:
        return self._flops['ADD']

    def reduce_count(self) -> int:
        return prod(self.inputs[0].shape) // prod(self.outputs[0].shape)
