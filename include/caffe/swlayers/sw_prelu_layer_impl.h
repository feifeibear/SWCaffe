#ifndef SW_RELU_LAYER_IMPL_H_
#define SW_RELU_LAYER_IMPL_H_

void sw_prelu_forward_impl_f(
        const float* in,
        float* out,
        float* slope_data,
        int count,
        int dim,
        int channels,
        int div_factor);

void sw_prelu_forward_impl_d(
        const double* in,
        double* out,
        double* slope_data,
        int count,
        int dim,
        int channels,
        int div_factor);

void sw_prelu_backward_impl_f(
        const float* bottom_data,
        const float* top_diff,
        float* bottom_diff,
        float* slope_data,
        float* slope_diff,
        int count,
        int dim,
        int channels,
        int div_factor,
        int param_propagate_down,
        int propagate_down);

void sw_prelu_backward_impl_d(
        const double* bottom_data,
        const double* top_diff,
        double* bottom_diff,
        double* slope_data,
        double* slope_diff,
        int count,
        int dim,
        int channels,
        int div_factor,
        int param_propagate_down,
        int propagate_down);

#endif
