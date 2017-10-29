from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Sqrt(function_node.FunctionNode):
    """Sqrt based on FunctionNode.

    note: not for more than 2-times differential,
          because div functions based on FunctionNode
          have not been implemented yet.
    """

    @property
    def label(self):
        return 'sqrt'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        self.retain_outputs((0,))
        return utils.force_array(xp.sqrt(x)),

    def backward(self, indexes, gy):
        # incomplete
        y, = self.get_retained_outputs()
        xp = cuda.get_array_module(y.data)
        if xp.any(y.data == 0.):
            return gy[0] / (y * 2. + 1e-6),
        else:
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
