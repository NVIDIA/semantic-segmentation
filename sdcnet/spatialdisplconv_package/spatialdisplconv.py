from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import spatialdisplconv_cuda

class SpatialDisplConvFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, input3, input4, kernel_size = 1):
        assert input1.is_contiguous(), "spatialdisplconv forward - input1 is not contiguous"
        assert input2.is_contiguous(), "spatialdisplconv forward - input2 is not contiguous"
        assert input3.is_contiguous(), "spatialdisplconv forward - input3 is not contiguous"
        assert input4.is_contiguous(), "spatialdisplconv forward - input4 is not contiguous"

        ctx.save_for_backward(input1, input2, input3, input4)
        ctx.kernel_size = kernel_size

        _, image_channels, _, _ = input1.size()
        batch_size, _, height, width = input2.size()
        output = input1.new(batch_size, image_channels, height, width).zero_()

        spatialdisplconv_cuda.forward(
            input1,
            input2,
            input3,
            input4,
            output,
            kernel_size
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        input1, input2, input3, input4 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input2.new(input2.size()).zero_())
        grad_input3 = Variable(input3.new(input3.size()).zero_())
        grad_input4 = Variable(input4.new(input4.size()).zero_())

        spatialdisplconv_cuda.backward(
            input1,
            input2,
            input3,
            input4,
            grad_output.data,
            grad_input1.data,
            grad_input2.data,
            grad_input3.data,
            grad_input4.data,
            ctx.kernel_size
        )

        return grad_input1, grad_input2, grad_input3, grad_input4, None

class SpatialDisplConv(Module):
    def __init__(self, kernel_size = 1):
        super(SpatialDisplConv, self).__init__()
        self.kernel_size = kernel_size


    def forward(self, input1, input2, input3, input4):

        return SpatialDisplConvFunction.apply(input1, input2, input3, input4, self.kernel_size)
