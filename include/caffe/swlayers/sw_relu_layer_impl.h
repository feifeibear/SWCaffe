#ifndef SW_RELU_LAYER_IMPL_H_
#define SW_RELU_LAYER_IMPL_H_

void sw_relu_forward_impl_f(
        const float* in,
        float* out,
        float negative_slope,
        int count);

void sw_relu_forward_impl_d(
        const double* in,
        double* out,
        double negative_slope,
        int count);

void sw_relu_backward_impl_f(
        const float* bottom_data,
        const float* top_diff,
        float* bottom_diff,
        float negative_slope,
        int count);

void sw_relu_backward_impl_d(
        const double* bottom_data,
        const double* top_diff,
        double* bottom_diff,
        double negative_slope,
        int count);

#endif
