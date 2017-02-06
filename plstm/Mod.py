import chainer
from chainer.utils import type_check
from chainer import utils
class Mod(chainer.function.Function) :

    @property
    def label(self) :
        return '__mod__'

    def check_type_forward(self, in_types) :
        type_check.expect(in_types.size() == 2)
        #import pdb; pdb.set_trace()
        type_check.expect(in_types[0].dtype.kind == in_types[1].dtype.kind,in_types[0].shape == in_types[1].shape)

    def forward(self, x) :
        return utils.force_array(x[0] % x[1]),

    def backward(self, x, gy) :
        return gy[0], -(x[0] // x[1])*gy[0]

def mod(x, y) :
    return Mod()(x, y)
