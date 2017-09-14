#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"
extern "C" {
  #include "caffe/swlayers/sw_softmax_layer_impl.h"
}
namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void mpe_softmax_forward_impl(
    const Dtype* bottom_data,
    const Dtype* sum_multiplier_,
    Dtype* scale_data,
    Dtype* top_data,
    int channels,
    int dim,
    int outer_num_,
    int inner_num_) {
  Dtype* bottom_data_T = (Dtype*)malloc(sizeof(Dtype)*outer_num_*dim);
  Dtype* top_data_T = (Dtype*)malloc(sizeof(Dtype)*outer_num_*dim);

  int i, k ,j;

  // matrix trans
  for(i=0; i < outer_num_;++i) {
    for(j=0;j < channels;++j) {
      for(k=0;k < inner_num_;++k) {
        bottom_data_T[i*dim+k*channels+j] = bottom_data[i*dim+j*inner_num_+k];
      }
    }
  }

  Dtype local_scale, stemp;
  for(i=0;i<outer_num_;++i) {
    for(j=0;j<inner_num_;++j) {
      local_scale = bottom_data_T[i*dim+j*channels];
      for(k=1;k<channels;++k) {
        local_scale = std::max(local_scale,bottom_data_T[i*dim+j*channels+k]);
      }
      stemp = 0.0;
      for(k=0;k<channels;++k) {
        top_data_T[i*dim+j*channels+k] = exp(bottom_data_T[i*dim+j*channels+k] -
                sum_multiplier_[k]*local_scale);
        stemp += top_data_T[i*dim+j*channels+k]*sum_multiplier_[k];
      }
      for(k=0;k<channels;++k) {
        top_data_T[i*dim+j*channels+k] = top_data_T[i*dim+j*channels+k]/stemp;
      }
    }
  }

  // matrix trans back
  for(i=0; i < outer_num_;++i) {
    for(j=0;j < channels;++j) {
      for(k=0;k < inner_num_;++k) {
        top_data[i*dim+j*inner_num_+k] = top_data_T[i*dim+k*channels+j];
      }
    }
  }

  free(bottom_data_T);
  free(top_data_T);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
#ifdef USE_SWSOFTMAX
  if(inner_num_>=64) {
  // sw softmax
  if(typeid(Dtype)==typeid(float)) {
    sw_softmax_forward_impl_f(
            (float*)bottom_data,
            (float*)sum_multiplier_.cpu_data(),
            (float*)scale_data,
            (float*)top_data,
            channels,
            dim,
            outer_num_,
            inner_num_);
  } else if(typeid(Dtype)==typeid(double)) {
    printf("Not implemented.\n");
    exit(0);
  }
  } else {
    // use MPE version
    mpe_softmax_forward_impl<Dtype>(
        bottom_data,
        sum_multiplier_.cpu_data(),
        scale_data,
        top_data,
        channels,
        dim,
        outer_num_,
        inner_num_);
  }
#else
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
#endif
  //bottom[0]->fjr_print_data();
  //top[0]->fjr_print_data();
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;

  //DLOG(INFO) << "fjrdebug : dim " << dim << " channels " << channels << " outer_num_ " <<  outer_num_;

  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);

  //top[0]->fjr_print_data();
  //top[0]->fjr_print_diff();
  //bottom[0]->fjr_print_diff();
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
