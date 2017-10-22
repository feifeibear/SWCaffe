#include <vector>
#include <assert.h>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#define USE_SWDNN
//#define TEST
//#ifdef USE_SWDNN

extern "C" {
#include "caffe/swlayers/sw_conv_layer_impl.h"
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
//#ifdef SW4CG
//template <typename Dtype>
//void ConvolutionLayer<Dtype>::Forward_cpu_4cg(const vector<Blob<Dtype>*>& bottom,
      //const vector<Blob<Dtype>*>& top) {
   ////const Dtype* weight = this->blobs_[0]->cpu_data();
   ////for (int i = 0; i < bottom.size(); ++i) {
      ////const Dtype* bottom_data = bottom[i]->cpu_data();
      ////Dtype* top_data = top[i]->mutable_cpu_data();
      ////for (int n = 0; n < this->num_; ++n) {
        ////this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            ////top_data + n * this->top_dim_);
        ////if (this->bias_term_) {
          ////const Dtype* bias = this->blobs_[1]->cpu_data();
          ////this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
        ////}
      ////}
    ////}
    //Sync_4cg((Dtype)0);
    //int threadidx = Caffe::solver_cgid();
    //CHECK_EQ((this->num_%NThread), 0);
    //int split_num_ = this->num_/NThread;
    //int num_offset_ = this->num_/NThread*threadidx;
  ////if(threadidx = 0){
    //const Dtype* weight = this->blobs_[0]->cpu_data();
    //for (int i = 0; i < bottom.size(); ++i) {
      //const Dtype* bottom_data = bottom[i]->cpu_data();
      //Dtype* top_data = top[i]->mutable_cpu_data();
//#ifdef DEBUG_VERBOSE_8
      //LOG(INFO) << "Rank " << Caffe::solver_rank() 
        //<<", CG "<<threadidx
        //<<": Do conv on batch idx ["<<num_offset_<<" "<<num_offset_+split_num_<<")";
//#endif
      //for (int n = 0; n < split_num_; ++n) {
//#ifdef DEBUG_VERBOSE_8
        //struct timeval ts, te;
        //gettimeofday(&ts, NULL);
//#endif
        //this->forward_cpu_gemm(bottom_data + (num_offset_+ n) * this->bottom_dim_, 
            //weight,
            //top_data + (num_offset_+n) * this->top_dim_);
//#ifdef DEBUG_VERBOSE_8
        //gettimeofday(&te, NULL);
        //double time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
        ////LOG_IF(INFO, Caffe::root_solver()) << "Root: \t\t" << i
        //LOG(INFO) << "Rank " << Caffe::solver_rank() 
          //<<", CG "<<Caffe::solver_cgid()
          //<< " : " << i
          //<< "th bottom " << n << "th iter forward_cpu_gemm in Forward_cpu cost time: " << time << "s";
        //gettimeofday(&ts, NULL);
//#endif
        //if (this->bias_term_) {
          //const Dtype* bias = this->blobs_[1]->cpu_data();
          //this->forward_cpu_bias(top_data + (num_offset_+n) * this->top_dim_, bias);
        //}
//#ifdef DEBUG_VERBOSE_4
        //gettimeofday(&te, NULL);
        //time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
        ////LOG_IF(INFO, Caffe::root_solver()) << "Root: \t\t" << i
        //LOG(INFO) << "Rank " << Caffe::solver_rank() 
          //<<", CG "<<threadidx
          //<< " : " << i
          //<< "th bottom " << n << "th iter forward_cpu_bias in Forward_cpu cost time: " << time << "s";
//#endif
      //}
    //}
    //Sync_4cg((Dtype)0);
//}
//#endif
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
#ifdef TEST
    Blob<Dtype> my_bottom_blob;
    Blob<Dtype> my_top_blob;
    Blob<Dtype> my_weight_blob;
    Blob<Dtype> my_bias_blob;

    const Dtype* myweight;
    Dtype* myweight_diff;
    const Dtype* mybottom_data;
    Dtype* mybottom_diff;
    Dtype* mytop_diff;
    Dtype* mybias_diff;

    for (int i = 0; i < top.size(); ++i) {

      my_bottom_blob.CopyFrom(*bottom[i], true, true);
      my_bottom_blob.CopyFrom(*bottom[i], false, true);
      my_top_blob.CopyFrom(*top[i], true, true);

      my_weight_blob.CopyFrom(*(this->blobs_[0]), false, true);
      my_weight_blob.CopyFrom(*(this->blobs_[0]), true, true);
      DLOG(INFO) << "sum of weight " << this->blobs_[0]->asum_data(); 
      DLOG(INFO) << "sum of my weight " << my_weight_blob.asum_data(); 
      my_bias_blob.CopyFrom(*(this->blobs_[1]), true, true);

      const Dtype* myweight       = my_weight_blob.cpu_data();
      Dtype* myweight_diff        = my_weight_blob.mutable_cpu_diff();
      const Dtype* mybottom_data  = my_bottom_blob.cpu_data();
      Dtype* mybottom_diff        = my_bottom_blob.mutable_cpu_diff();
      Dtype* mytop_diff           = my_top_blob.mutable_cpu_diff();
      Dtype* mybias_diff          = my_bias_blob.mutable_cpu_diff();

      DLOG(INFO) << "begin my code";

      int mypad = 0;
      const int* pad_data = this->pad_.cpu_data();
      if(this->num_spatial_axes_)
        mypad = pad_data[0];

      if (this->param_propagate_down_[0] || propagate_down[i]) {
        if(sizeof(Dtype)== sizeof(double))
		{
			 conv_backward_pad_impl_d(
			  //const Type* in,
			  (double*)mybottom_data,
			  //const Type* out_grad,
			  (double*)mytop_diff,
			  //Type* weight,
			  (double*)myweight,
			  //Type* in_grad,
			  (double*)mybottom_diff,
			  //Type* weight_diff,
			  (double*)myweight_diff,
			  //Type* bias_grad,
			  //mybias_diff,
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
			conv_backward_pad_impl_f(
			  //const Type* in,
			  (float*)mybottom_data,
			  //const Type* out_grad,
			  (float*)mytop_diff,
			  //Type* weight,
			  (float*)myweight,
			  //Type* in_grad,
			  (float*)mybottom_diff,
			  //Type* weight_diff,
			  (float*)myweight_diff,
			  //Type* bias_grad,
			  //mybias_diff,
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
	}//for
    DLOG(INFO) << "backward OK";
#endif

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
#ifdef TEST

  DLOG(INFO) << "bottom_diff size " << bottom[0]->count();
  DLOG(INFO) << "bottom_diff size " << my_bottom_blob.count();
  assert( bottom[0]->count() == my_bottom_blob.count() );
  for(int i = 0; i < bottom[0]->count(); ++i){
    if( fabs(bottom[0]->cpu_diff()[i] - mybottom_diff[i]) > 1e-4)
      DLOG(INFO) << i << " : bottom diff caffe : " << bottom[0]->cpu_diff()[i]
        << " my : " << mybottom_diff[i];
  }
  DLOG(INFO) << "CHECK BACK bottom_diff OK";

  DLOG(INFO) << "weight diff size " << this->blobs_[0]->count();
  assert( this->blobs_[0]->count() == my_weight_blob.count() );
  for(int i = 0; i < this->blobs_[0]->count(); ++i){
    if( fabs(this->blobs_[0]->cpu_diff()[i] - myweight_diff[i]) > 1e-4)
      DLOG(INFO) << "bottom diff caffe : " << this->blobs_[0]->cpu_diff()[i]
        << " my : " << myweight_diff[i];
  }
  DLOG(INFO) << "CHECK BACK weight OK";

  DLOG(INFO) << "bias diff size " << this->blobs_[0]->count();
  assert( this->blobs_[1]->count() == my_bias_blob.count() );
  for(int i = 0; i < this->blobs_[1]->count(); ++i){
    if( fabs(this->blobs_[1]->cpu_diff()[i] - mybias_diff[i]) > 1e-4)
      DLOG(INFO) << "bottom diff caffe : " << this->blobs_[1]->cpu_diff()[i]
        << " my : " << mybias_diff[i];
  }
  DLOG(INFO) << "CHECK BACK bias OK";
  times++;
  if( times == 10 )
    exit(0);
#endif
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
