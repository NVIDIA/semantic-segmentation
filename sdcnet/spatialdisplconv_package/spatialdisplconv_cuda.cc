#include <ATen/ATen.h>
#include <torch/torch.h>

#include "spatialdisplconv_kernel.cuh"

int spatialdisplconv_cuda_forward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& input3,
    at::Tensor& input4,
    at::Tensor& output, 
    int kernel_size) {

        spatialdisplconv_kernel_forward(
            input1,
            input2,
            input3,
            input4,
            output,
            kernel_size
        );

        return 1;
}


int spatialdisplconv_cuda_backward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& input3,
    at::Tensor& input4,
    at::Tensor& gradOutput, 
    at::Tensor& gradInput1, 
    at::Tensor& gradInput2,
    at::Tensor& gradInput3,
    at::Tensor& gradInput4,
    int kernel_size

) {
    spatialdisplconv_kernel_backward(
        input1,
        input2,
        input3,
        input4,
        gradOutput, 
        gradInput1,
        gradInput2,
        gradInput3,
        gradInput4,
        kernel_size
    );

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spatialdisplconv_cuda_forward, "SpatialDisplConv forward (CUDA)");
  m.def("backward", &spatialdisplconv_cuda_backward, "SpatialDisplConv backward (CUDA)");
}
