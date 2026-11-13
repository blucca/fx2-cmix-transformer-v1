"""Build one static-M1, symmetric Q8/block32 operator for the native library."""
import numpy as np
from onnx import TensorProto, helper, numpy_helper


def native_m1(b, s):
    if b.dtype != np.uint8 or b.ndim != 3 or b.shape[2] != 32:
        raise ValueError('B must be uint8 [N,K/32,32]')
    n, kb, _ = b.shape
    if n <= 0 or n % 16 or kb <= 0:
        raise ValueError('N must be a positive multiple of 16; K a positive multiple of 32')
    if s.dtype != np.float32 or s.shape != (n * kb,) or not np.isfinite(s).all():
        raise ValueError('S must be finite float32 [N*K/32]')
    k = kb * 32
    node = helper.make_node('Q8MatMulM1', ['A', 'B', 'S'], ['Y'],
                            domain='org.blucca', K=k, N=n)
    graph = helper.make_graph([node], 'q8-m1',
        [helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, k])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, n])],
        [numpy_helper.from_array(b, 'B'), numpy_helper.from_array(s, 'S')])
    model = helper.make_model(graph,
        opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('org.blucca', 1)],
        ir_version=9)
    return model.SerializeToString(), b.nbytes + s.nbytes
