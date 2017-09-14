#ifndef SW_SOFTMAX_LAYER_IMPL_H_
#define SW_SOFTMAX_LAYER_IMPL_H_
void sw_softmax_forward_impl_f(
    const float* bottom_data,
    const float* sum_multiplier_,
    float* scale_data,
    float* top_data,
    int channels,
    int dim,
    int outer_num_,
    int inner_num_);
#endif
