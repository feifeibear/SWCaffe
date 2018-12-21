#include <algorithm>
#include <vector>
#include <assert.h>
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"


extern "C"{
    #include "swbatchnorm.h"
}

namespace caffe {


static inline unsigned long rpcc()
{
        unsigned long time;
        asm("rtc %0": "=r" (time) : );
        return time;
}



template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the bias correction to zero.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } else {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_SWBATCHNORM

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int num = bottom[0]->shape(0);
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

    if (bottom[0] != top[0]) {
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }

    if(spatial_dim < 60 * 60)
    {

      if(use_global_stats_)
      {


          if(sizeof(Dtype)==sizeof(float)){
            sw_batch_norm_use_forward_impl_f(
                (float*)bottom_data,
                (float*)top_data,
                (float*)this->blobs_[0]->mutable_cpu_data(),
                (float*)this->blobs_[1]->mutable_cpu_data(),
                (float*)this->blobs_[2]->mutable_cpu_data(),
                (float*)mean_.mutable_cpu_data(),
                (float*)variance_.mutable_cpu_data(),
                (float*)temp_.mutable_cpu_data(),
                (float*)x_norm_.mutable_cpu_data(),
                eps_,
                num,         //batch_size
                channels_,   //C
                spatial_dim  //H*W
            );
          }else{
            sw_batch_norm_use_forward_impl_d(
                (double*)bottom_data,
                (double*)top_data,
                (double*)this->blobs_[0]->mutable_cpu_data(),
                (double*)this->blobs_[1]->mutable_cpu_data(),
                (double*)this->blobs_[2]->mutable_cpu_data(),
                (double*)mean_.mutable_cpu_data(),
                (double*)variance_.mutable_cpu_data(),
                (double*)temp_.mutable_cpu_data(),
                eps_,
                num,         //batch_size
                channels_,   //C
                spatial_dim  //H*W
            );
          }
      }
      else
      {
          if(sizeof(Dtype)==sizeof(float)){
          sw_batch_norm_nouse_forward_impl_f(
              (float*)bottom_data,
              (float*)top_data,
              (float*)num_by_chans_.mutable_cpu_data(),
              (float*)temp_.mutable_cpu_data(),
              (float*)this->blobs_[0]->mutable_cpu_data(),
              (float*)this->blobs_[1]->mutable_cpu_data(),
              (float*)this->blobs_[2]->mutable_cpu_data(),
              (float*)(&moving_average_fraction_),
              (float*)x_norm_.mutable_cpu_data(),
              eps_,
              num,
              channels_,
              spatial_dim
          );
          }
          else{
          sw_batch_norm_nouse_forward_impl_d(
              (double*)bottom_data,
              (double*)top_data,
              (double*)num_by_chans_.mutable_cpu_data(),
              (double*)temp_.mutable_cpu_data(),
              (double*)this->blobs_[0]->mutable_cpu_data(),
              (double*)this->blobs_[1]->mutable_cpu_data(),
              (double*)this->blobs_[2]->mutable_cpu_data(),
              (double*)(&moving_average_fraction_),
              eps_,
              num,
              channels_,
              spatial_dim
          );
          }

      }

    }
    else  //spatial_dim >= 60*60
    {
      if (use_global_stats_) {
        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
            0 : 1 / this->blobs_[2]->cpu_data()[0];
        caffe_cpu_scale(variance_.count(), scale_factor,
            this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
        caffe_cpu_scale(variance_.count(), scale_factor,
            this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
      } else {
        // compute mean
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), bottom_data,
            spatial_sum_multiplier_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
            mean_.mutable_cpu_data());
      }

      // subtract mean
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
          batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
          num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
          spatial_dim, 1, -1, num_by_chans_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), 1., top_data);

      if (!use_global_stats_) {
        // compute variance using var(X) = E((X-EX)^2)
        caffe_sqr<Dtype>(top[0]->count(), top_data,
                         temp_.mutable_cpu_data());  // (X-EX)^2
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), temp_.cpu_data(),
            spatial_sum_multiplier_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
            variance_.mutable_cpu_data());  // E((X_EX)^2)

        // compute and save moving average
        this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
            moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
        int m = bottom[0]->count()/channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
        caffe_cpu_axpby(variance_.count(), bias_correction_factor,
            variance_.cpu_data(), moving_average_fraction_,
            this->blobs_[1]->mutable_cpu_data());
      }

      // normalize variance
      caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
      caffe_sqrt(variance_.count(), variance_.cpu_data(),
                 variance_.mutable_cpu_data());

      // replicate variance to input size
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
          batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
          num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
          spatial_dim, 1, 1., num_by_chans_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
      caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
      // TODO(cdoersch): The caching is only needed because later in-place layers
      //                 might clobber the data.  Can we skip this if they won't?
      caffe_copy(x_norm_.count(), top_data,
          x_norm_.mutable_cpu_data());
    }



#else
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    // compute mean
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        mean_.mutable_cpu_data());
  }

  // subtract mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_sqr<Dtype>(top[0]->count(), top_data,
                     temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), temp_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_cpu_axpby(variance_.count(), bias_correction_factor,
        variance_.cpu_data(), moving_average_fraction_,
        this->blobs_[1]->mutable_cpu_data());
  }

  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_sqrt(variance_.count(), variance_.cpu_data(),
             variance_.mutable_cpu_data());

  // replicate variance to input size
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data,
      x_norm_.mutable_cpu_data());
#endif
  
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {


#ifdef USE_SWBATCHNORM
    const Dtype* top_data = x_norm_.cpu_data();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
    const Dtype* top_diff;
    if(bottom[0]!=top[0]){
        top_diff=top[0]->cpu_diff();
    }else{
        caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
        top_diff = x_norm_.cpu_diff();
    }
    Dtype* bottom_diff=bottom[0]->mutable_cpu_diff();
    
    if(spatial_dim < 60*60)
    {

      if(use_global_stats_)
      {
          if(sizeof(Dtype)==sizeof(float)){
          sw_batch_norm_use_backward_impl_f(
              temp_.count(),
              num,
              channels_,
              spatial_dim,
              (float*)top_diff,
              (float*)temp_.cpu_data(),
              (float*)bottom_diff
          );
          }
          else{
          sw_batch_norm_use_backward_impl_d(
              temp_.count(),
              (double*)top_diff,
              (double*)temp_.cpu_data(),
              (double*)bottom_diff
          );
          }
          return;
      }
      if(sizeof(Dtype)==sizeof(float)){
      sw_batch_norm_nouse_backward_impl_f(
          (float*)top_data,
          (float*)top_diff,
          (float*)bottom_diff,
          num,
          channels_,
          spatial_dim,
          temp_.count(),
          (float*)temp_.cpu_data()
      );
      }
      else{
      sw_batch_norm_nouse_backward_impl_d(
          (double*)top_data,
          (double*)top_diff,
          (double*)bottom_diff,
          num,
          channels_,
          spatial_dim,
          temp_.count(),
          (double*)temp_.cpu_data()
      ); 
      }

    } //spatial_dim >= 60*60
    else
    {
      if (use_global_stats_) {
        caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
        return;
      }
      caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
          bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
          num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
          num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
          mean_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
          batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
          num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
          spatial_dim, 1, 1., num_by_chans_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

      caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

      caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
          top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
          num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
          num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
          mean_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
          batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
          num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
          spatial_dim, 1, 1., num_by_chans_.cpu_data(),
          spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

      caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
          Dtype(-1. / (num * spatial_dim)), bottom_diff);

      caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
    }

#else
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
    top_diff = x_norm_.cpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (use_global_stats_) {
    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
    //LOG(INFO)<<"backward use global="<<use_global_stats_;
#endif


}


#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);
REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace caffe
