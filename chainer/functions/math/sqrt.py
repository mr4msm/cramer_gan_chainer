import numpy

from chainer import cuda
from chainer import function
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Sqrt(function_node.FunctionNode):

    @property
    def label(self):
        return 'sqrt'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        return utils.force_array(numpy.sqrt(x[0])),

    def forward_gpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        return cuda.cupy.sqrt(x[0]),

    def backward(self, indexes, gy):
        y = self.get_retained_outputs()[0]
        return gy[0] / (y * 2.),


def sqrt(x):
    """Elementwise square root function.

    .. math::
       y_i = \\sqrt x_i.

    If the value of :math:`x_i` is negative, it returns ``Nan`` for :math:`y_i`
    respect to underlying numpy and cupy specification.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Sqrt().apply((x,))[0]


def rsqrt(x):
    """Computes elementwise reciprocal of square root of input :math:`x_i`.

    .. math::
       y_i = {1 \\over \\sqrt x_i}.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :func:`~chainer.functions.sqrt`
    """
    return 1.0 / sqrt(x)
