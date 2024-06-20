import logging
import onnx

log = logging.getLogger(__name__)


def generate_roofline_test_model(n_start: int, expands: int, expands_memory: int = 0) -> onnx.ModelProto:
    """
    generate an onnx model includes matmul and relu to test hardware max FLOPS and MEM.BW (roofline)

    matrix size will be (n_start, n_start) to (n_start * 2^expands, n_start * 2^expands)
    the model also have batch_size dim to keep compatibility, but should set to 1
    """
    log.info("generating roofline_test_model, size from %d to %d", n_start, n_start * (2**expands))

    N = n_start
    input = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, ['batch_size', N, N])

    nodes = []
    initializer = []
    # initializer.append(onnx.helper.make_tensor('scale', onnx.TensorProto.FLOAT, [3], [1, 2, 2]))
    next_input = 'input'
    for i in range(expands):
        inner = f'out{i}'
        nodes.append(
            onnx.helper.make_node(
                "Transpose",                   # type
                [next_input],                  # inputs
                [inner+'_t'],                  # outputs
                f"Transpose_{N}",              # name
                perm=[0, 2, 1],
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Relu",                        # type
                [inner+'_t'],                  # inputs
                [inner+'_r'],                  # outputs
                f"Relu_{N}",                   # name
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "MatMul",                   # type
                [inner+'_r', next_input],   # inputs
                [inner],                    # outputs
                f"MatMul_{N}",              # name
            )
        )
        # nodes.append(
        #     onnx.helper.make_node(
        #         "Add",                        # type
        #         [inner, 'one'],                  # inputs
        #         [inner+'_a'],                  # outputs
        #         f"Add_{N}",                   # name
        #     )
        # )
        # nodes.append(
        #     onnx.helper.make_node(
        #         "Concat",                   # type
        #         [inner, inner],       # inputs
        #         [inner+'_c1'],              # outputs
        #         f"Concat_{N}_1",        # name
        #         axis = 2,
        #     )
        # )
        # nodes.append(
        #     onnx.helper.make_node(
        #         "Concat",                   # type
        #         [inner+'_c1', inner+'_c1'],       # inputs
        #         [inner+'_c2'],              # outputs
        #         f"Concat_{N}_2",        # name
        #         axis = 1,
        #     )
        # )
        # next_input = inner+'_c2'
        # nodes.append(
        #     onnx.helper.make_node(
        #         "Resize",                   # type
        #         [inner, '', 'scale'],       # inputs
        #         [inner+'_2x'],              # outputs
        #         f"Resize_{N}to{2*N}",       # name
        #     )
        # )
        nodes.append(
            onnx.helper.make_node(
                "Concat",                   # type
                [inner, inner],             # inputs
                [inner+'_2x_1'],              # outputs
                f"Concat_{N}to{2*N}_1",       # name
                axis=1,
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Concat",                   # type
                [inner+'_2x_1', inner+'_2x_1'],             # inputs
                [inner+'_2x'],              # outputs
                f"Concat_{N}to{2*N}_2",       # name
                axis=2,
            )
        )
        next_input = inner+'_2x'

        N *= 2

    for i in range(expands, expands + expands_memory):
        inner = f'out{i}'
        nodes.append(
            onnx.helper.make_node(
                "Transpose",                   # type
                [next_input],                  # inputs
                [inner+'_t'],                  # outputs
                f"Transpose_{N}",              # name
                perm=[0, 2, 1],
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Relu",                        # type
                [inner+'_t'],                  # inputs
                [inner+'_r'],                  # outputs
                f"Relu_{N}",                   # name
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Add",                      # type
                [inner+'_r', next_input],   # inputs
                [inner],                    # outputs
                f"Add_{N}",                 # name
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Concat",                       # type
                [inner, inner],                 # inputs
                [inner+'_2x_1'],                # outputs
                f"Concat_{N}to{2*N}_1",         # name
                axis=1,
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Concat",                       # type
                [inner+'_2x_1', inner+'_2x_1'], # inputs
                [inner+'_2x'],                  # outputs
                f"Concat_{N}to{2*N}_2",         # name
                axis=2,
            )
        )
        next_input = inner+'_2x'

        N *= 2

    output = onnx.helper.make_tensor_value_info(next_input, onnx.TensorProto.FLOAT, ['batch_size', N, N])

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes,              # nodes
        "test-model",       # name
        [input],            # inputs
        [output],           # outputs
        initializer
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="proof")
    onnx.checker.check_model(model_def)

    return model_def
