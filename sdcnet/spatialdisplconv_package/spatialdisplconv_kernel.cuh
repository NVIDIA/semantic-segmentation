#pragma once

#include <ATen/ATen.h>

void spatialdisplconv_kernel_forward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& input3,
    at::Tensor& input4,
    at::Tensor& output, 
    int kernel_size
);

void spatialdisplconv_kernel_backward(
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

);
