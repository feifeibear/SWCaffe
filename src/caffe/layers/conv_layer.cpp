#include <vector>
#include <assert.h>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#define USE_SWDNN
//#define TEST
//#ifdef USE_SWDNN

extern "C" {
#include "caffe/swlayers/sw_conv_layer_impl.h"
#ifdef SW4CG
#include "caffe/util/sw_memcpy.h"
#endif
}
//#endif

static int times = 0;
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}
#ifdef SW4CG
#ifdef SW4CG_CONV_FW
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu_4cg(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    Sync_4cg((Dtype)0);
    int cgid = Caffe::solver_cgid();
    CHECK_EQ((this->num_%NThread), 0);
    int split_num_ = this->num_/NThread;
    int num_offset_ = this->num_/NThread*cgid;
    const Dtype* weight = this->blobs_[0]->cpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < split_num_; ++n) {
        this->forward_cpu_gemm_4cg(bottom_data + (num_offset_+ n) * this->bottom_dim_, 
            weight,
            top_data + (num_offset_+n) * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias_4cg(top_data + (num_offset_+n) * this->top_dim_, bias);
        }
      }
    }
    Sync_4cg((Dtype)0);
}
#endif
#ifdef SW4CG_CONV_BW
template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu_4cg(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  Sync_4cg((Dtype)0);
  int cgid = Caffe::solver_cgid();
  CHECK_EQ((this->num_%NThread), 0);
  int split_num_ = this->num_/NThread;
  int num_offset_ = this->num_/NThread*cgid;

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int weight_diff_size = this->blobs_[0]->count();

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      int bias_diff_size = this->blobs_[1]->count();
      //caffe_set(bias_diff_size, static_cast<Dtype>(0), this->tmp_bias_diff[cgid]);
      if(sizeof(Dtype) == sizeof(double)){
        sw_memcpy_d((double*)bias_diff, (double*)this->tmp_bias_diff[cgid], bias_diff_size);
      }else if(sizeof(Dtype)==sizeof(float)){
        sw_memcpy_f((float*)bias_diff, (float*)this->tmp_bias_diff[cgid], bias_diff_size);
      }else{
        memcpy(this->tmp_bias_diff[cgid], bias_diff, bias_diff_size*sizeof(Dtype));
      }
      for (int n = 0; n < split_num_; ++n) {
        this->backward_cpu_bias_4cg(this->tmp_bias_diff[cgid], top_diff + (num_offset_+n) * this->top_dim_);
      }
    }
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    if (this->param_propagate_down_[0]) {
      //caffe_set(weight_diff_size, static_cast<Dtype>(0), this->tmp_weight_diff[cgid]);
      if(sizeof(Dtype) == sizeof(double)){
        sw_memcpy_d((double*)weight_diff, (double*)this->tmp_weight_diff[cgid], weight_diff_size);
      }else if(sizeof(Dtype)==sizeof(float)){
        sw_memcpy_f((float*)weight_diff, (float*)this->tmp_weight_diff[cgid], weight_diff_size);
      }else{
        memcpy(this->tmp_weight_diff[cgid], weight_diff, weight_diff_size*sizeof(Dtype));
      }
      for (int n = 0; n < split_num_; ++n) {
        this->weight_cpu_gemm_4cg(bottom_data + (num_offset_+n) * this->bottom_dim_,
            top_diff + (num_offset_+n) * this->top_dim_, this->tmp_weight_diff[cgid]);
      }
    }
    // gradient w.r.t. bottom data, if necessary.
    if (propagate_down[i]) {
      for (int n = 0; n < split_num_; ++n) {
        this->backward_cpu_gemm_4cg(top_diff + (num_offset_+n) * this->top_dim_, weight,
            bottom_diff + (num_offset_+n) * this->bottom_dim_);
      }
    }
  }
  Sync_4cg((Dtype)0);
  if(cgid==0){
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      int bias_diff_size = this->blobs_[1]->count();
      for(int j=0; j<NThread; j++){
        caffe_axpy<Dtype>(bias_diff_size, Dtype(1.), this->tmp_bias_diff[j], bias_diff);
      }
    }
    if (this->param_propagate_down_[0]) {
      for(int j=0; j<NThread; j++){
        caffe_axpy<Dtype>(weight_diff_size, Dtype(1.), this->tmp_weight_diff[j], weight_diff);
      }
    }
  }
}
#endif
#endif

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifdef USE_SWDNN
  //assert(typeid(Dtype) == typeid(double));
  const Dtype* weight       = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const int* stride_data = this->stride_.cpu_data();
    const int* pad_data = this->pad_.cpu_data();
    int mypad = 0;
    if(this->num_spatial_axes_)
      mypad = pad_data[0];

    const int* dilation_data = this->dilation_.cpu_data();
    if(bottom[0]->num() >= 128 
        && bottom[0]->channels() >= 64 && bottom[0]->channels() % 32 == 0 
        && top[0]->channels() >= 64 && top[0]->channels() % 32 == 0
        && this->group_==1
      ){
#ifdef DEBUG_VERBOSE_SWDNN
    DLOG(INFO)<<"DEBUG FORWARD CONV 3";
#endif
      const Dtype* bottom_data  = bottom[i]->cpu_data();
      Dtype* top_data           = top[i]->mutable_cpu_data();

      if(sizeof(Dtype) == sizeof(double))
      {
        //sw_conv_forward_impl_d(
        sw_conv_forward_pad_impl_d(
            (double*)bottom_data,
            (double*)weight,
            (double*)top_data,
            //bias_data,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            //int pad
            mypad
            );
      }
      else 
      {

        sw_conv_forward_pad_impl_f(
            (float*)bottom_data,
            (float*)weight,
            (float*)top_data,
            //bias_data,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            //int pad
            mypad
            );
      }
    }

    else {
#ifdef DEBUG_VERBOSE_SWDNN
    DLOG(INFO)<<"DEBUG FORWARD CONV 4";
#endif
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(bottom_data
            + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
      }
    }

    if (this->bias_term_) {

      Dtype* top_data = top[i]->mutable_cpu_data();
      const Dtype* bias = this->blobs_[1]->cpu_data();
      for (int n = 0; n < this->num_; ++n)
        this->forward_cpu_bias(top_data
            + n * this->top_dim_, bias);
    }
  }//for
#else
   const Dtype* weight = this->blobs_[0]->cpu_data();
   for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    }
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#ifdef USE_SWDNN
    //assert(typeid(Dtype) == typeid(double));
    const Dtype* weight    = this->blobs_[0]->cpu_data();
    Dtype* weight_diff     = this->blobs_[0]->mutable_cpu_diff();

      int mypad = 0;
      const int* pad_data = this->pad_.cpu_data();
      if(this->num_spatial_axes_)
        mypad = pad_data[0];

    for (int i = 0; i < top.size(); ++i) {
      const Dtype* bottom_data  = bottom[i]->cpu_data();
      Dtype* bottom_diff        = bottom[i]->mutable_cpu_diff();
      const Dtype* top_diff     = top[i]->mutable_cpu_diff();

        if (this->bias_term_ && this->param_propagate_down_[1]) {
          Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
          for (int n = 0; n < this->num_; ++n) {
            this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
          }
        }


    if(    bottom[0]->num() >= 64 && bottom[0]->num() % 32 == 0 
        && bottom[0]->channels() >= 128 && bottom[0]->channels()%128 == 0
        && top[0]->channels() >= 64 && top[0]->channels() % 32 == 0 
        && this->group_==1){

      if (this->param_propagate_down_[0] || propagate_down[i]) {
        if(sizeof(Dtype)== sizeof(double))
    {
          sw_conv_backward_pad_impl_d(
            //const Type* in,
            (double*)bottom_data,
            //const Type* out_grad,
            (double*)top_diff,
            //Type* weight,
            (double*)weight,
            //Type* in_grad,
            (double*)bottom_diff,
            //Type* weight_diff,
            (double*)weight_diff,
            //Type* bias_grad,
            //bias_diff,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            mypad
            );
    }
    else 
    {
          sw_conv_backward_pad_impl_f(
            //const Type* in,
            (float*)bottom_data,
            //const Type* out_grad,
            (float*)top_diff,
            //Type* weight,
            (float*)weight,
            //Type* in_grad,
            (float*)bottom_diff,
            //Type* weight_diff,
            (float*)weight_diff,
            //Type* bias_grad,
            //bias_diff,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            mypad
            );
    }
        }
    }
    else {
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }//else
  }//for
#else

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
#endif
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
