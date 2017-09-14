#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
extern "C" {
#include "caffe/swlayers/sw_relu_layer_impl.h"
}
namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_SWRELU
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  if( typeid(Dtype) == typeid(double) ) {
    sw_relu_forward_impl_d(
        (double*)bottom_data,
        (double*)top_data,
        (double) negative_slope,
        // int count
        count
        );
  } else if ( typeid(Dtype) == typeid(float) ) {
    sw_relu_forward_impl_f(
        (float*)bottom_data,
        (float*)top_data,
        (float)negative_slope,
        // int count
        count
        );
  }
#else
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
#endif
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
#ifdef USE_SWRELU
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    if( typeid(Dtype) == typeid(double) ) {
      sw_relu_backward_impl_d(
        (double*)bottom_data,
        (double*)top_diff,
        (double*)bottom_diff,
        (double) negative_slope,
        // int count
        count
        );
    } else if ( typeid(Dtype) == typeid(float) ) {
      sw_relu_backward_impl_f(
        (float*)bottom_data,
        (float*)top_diff,
        (float*)bottom_diff,
        (float)negative_slope,
        // int count
        count
        );
    }
  }
#else
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
#endif
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
