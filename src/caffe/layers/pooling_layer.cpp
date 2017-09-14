#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_SWPOOL
extern "C" {
#include "caffe/swlayers/sw_pool_layer_impl.h"
}
#endif

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  	
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);  // ---!!
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);  // ---!!
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);  // ---!!
		
    // The main loop
#ifdef USE_SWPOOL
  if(pooling_judge_condition(bottom[0]->num(),channels_,pooled_height_, pooled_width_) >0)				
	{
    //printf("Enter Pool Forward_cpu\n");
    if(sizeof(Dtype) == sizeof(double))
		   pooling_forward_max_d(bottom[0]->num(),channels_,(double*)top_data,(const double*)bottom_data,(int*)mask,(double*)top_mask,bottom[0]->offset(0, 1),
				top[0]->offset(0, 1),top.size() - 1,pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
		else
			pooling_forward_max_f(bottom[0]->num(),channels_,(float*)top_data,(const float*)bottom_data,(int*)mask,(float*)top_mask,bottom[0]->offset(0, 1),
				top[0]->offset(0, 1),top.size() - 1,pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
	}
	else
	{
		for (int n = 0; n < bottom[0]->num(); ++n) {
		  for (int c = 0; c < channels_; ++c) {
			for (int ph = 0; ph < pooled_height_; ++ph) {
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				int hstart = ph * stride_h_ - pad_h_;
				int wstart = pw * stride_w_ - pad_w_;
				int hend = min(hstart + kernel_h_, height_);
				int wend = min(wstart + kernel_w_, width_);
				hstart = max(hstart, 0);
				wstart = max(wstart, 0);
				const int pool_index = ph * pooled_width_ + pw;
				for (int h = hstart; h < hend; ++h) {
				  for (int w = wstart; w < wend; ++w) {
					const int index = h * width_ + w;
					if (bottom_data[index] > top_data[pool_index]) {
					  top_data[pool_index] = bottom_data[index];
					  if (use_top_mask) {
						top_mask[pool_index] = static_cast<Dtype>(index);
					  } else {
						mask[pool_index] = index;
					  }
					}
				  }
				}
			  }
			}
			// compute offset
			bottom_data += bottom[0]->offset(0, 1);
			top_data += top[0]->offset(0, 1);
			if (use_top_mask) {
			  top_mask += top[0]->offset(0, 1);
			} else {
			  mask += top[0]->offset(0, 1);
			}
		  }
		}
	}
 
#else
	for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                  //printf("find a max, index : %d, mask[%d] : %d\n", index, pool_index, mask[pool_index]);
                }
              }
            }
            if(mask[pool_index] < 0) {
              printf("Pool Forward Error : n: %d, c: %d, ph: %d, pw: %d, top_data[%d]: %d\n",
                n, c, ph, pw, pool_index, top_data[pool_index]);
              printf("mask[%d] : %d, width_ : %d, height_ : %d, pooled_height_ : %d, pooled_width_ : %d\n", 
                  pool_index, mask[pool_index], width_, height_, pooled_height_, pooled_width_);
              exit(0);
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
#endif
    break;
  case PoolingParameter_PoolMethod_AVE:
    
    // The main loop
    //printf("AVG=num=%d channels_=%d pooled_height_=%d pooled_width_=%d stride_h_=%d stride_w_=%d pad_h_=%d pad_w_=%d kernel_h_=%d kernel_w_=%d height_=%d width_=%d top_offset=%d bottom_offset=%d\n",\
    //    bottom[0]->num(),channels_,pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_,top[0]->offset(0,1), bottom[0]->offset(0,1));
#ifdef USE_SWPOOL
  if(pooling_judge_condition(bottom[0]->num(),channels_,pooled_height_, pooled_width_) >0)
	{
		if(sizeof(Dtype) == sizeof(double))	
		    pooling_forward_avg_d(bottom[0]->num(),channels_,(double*)top_data,(const double*)bottom_data,bottom[0]->offset(0, 1),
				top[0]->offset(0, 1),pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
		else
			pooling_forward_avg_f(bottom[0]->num(),channels_,(float*)top_data,(const float*)bottom_data,bottom[0]->offset(0, 1),
				top[0]->offset(0, 1),pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
	}
	else
	{
		for (int i = 0; i < top_count; ++i) {
          top_data[i] = 0;
        }
		for (int n = 0; n < bottom[0]->num(); ++n) {
		  for (int c = 0; c < channels_; ++c) {
			for (int ph = 0; ph < pooled_height_; ++ph) {
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				int hstart = ph * stride_h_ - pad_h_;
				int wstart = pw * stride_w_ - pad_w_;
				int hend = min(hstart + kernel_h_, height_ + pad_h_);
				int wend = min(wstart + kernel_w_, width_ + pad_w_);
				int pool_size = (hend - hstart) * (wend - wstart);
				hstart = max(hstart, 0);
				wstart = max(wstart, 0);
				hend = min(hend, height_);
				wend = min(wend, width_);
				for (int h = hstart; h < hend; ++h) {
				  for (int w = wstart; w < wend; ++w) {
					top_data[ph * pooled_width_ + pw] +=
						bottom_data[h * width_ + w];
				  }
				}
				top_data[ph * pooled_width_ + pw] /= pool_size;
			  }
			}
			// compute offset
			bottom_data += bottom[0]->offset(0, 1);
			top_data += top[0]->offset(0, 1);
		  }
		}
	}    
#else
	for (int i = 0; i < top_count; ++i) {
        top_data[i] = 0;
    }
	for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
#endif
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);  // ---!!
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;  
  /*
  Dtype dSum1=0,dSum2=0;
  Dtype* p_bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_ref = (Dtype*)malloc(bottom[0]->count()*sizeof(Dtype));
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff_ref);  // ---!!
  */
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
#ifdef USE_SWPOOL
  if(pooling_judge_condition(top[0]->num(),channels_,pooled_height_, pooled_width_) >0 )
	{
		//printf("Enter Pool Backward_cpu\n");
		if(	sizeof(Dtype) == sizeof(double))	
		   pooling_backward_max_d(top[0]->num(),channels_,(const double*)top_diff,(double*)bottom_diff,(const int*)mask,(const double*)top_mask,
				bottom[0]->offset(0, 1),top[0]->offset(0, 1),top.size() - 1,pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);	 
        else
            pooling_backward_max_f(top[0]->num(),channels_,(const float*)top_diff,(float*)bottom_diff,(const int*)mask,(const float*)top_mask,
				bottom[0]->offset(0, 1),top[0]->offset(0, 1),top.size() - 1,pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);				
	}
	else
	{
		for (int n = 0; n < top[0]->num(); ++n) {
		  for (int c = 0; c < channels_; ++c) {
			for (int ph = 0; ph < pooled_height_; ++ph) {
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				const int index = ph * pooled_width_ + pw;
				const int bottom_index =
					use_top_mask ? top_mask[index] : mask[index];
				bottom_diff[bottom_index] += top_diff[index];
			  }
			}
			bottom_diff += bottom[0]->offset(0, 1);
			top_diff += top[0]->offset(0, 1);
			if (use_top_mask) {
			  top_mask += top[0]->offset(0, 1);
			} else {
			  mask += top[0]->offset(0, 1);
			}
		  }
		}
	}  
    /*for(int i=0;i<bottom[0]->count();i++)
    {
		if(fabs(p_bottom_diff[i]-bottom_diff_ref[i]) >1e-4)
			printf("%15.3f vs %15.3f\n",p_bottom_diff[i],bottom_diff_ref[i]);
		dSum1 += p_bottom_diff[i];
		dSum2 += bottom_diff_ref[i];
	}		
	printf("backward pool dSum1=%15.3f dSum2=%15.3f\n",dSum1,dSum2);
	*/
 #else
	for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            if(bottom_index < 0){
              printf("Pool Error bottom_index %d, n: %d, c: %d, ph: %d, pw: %d, index: %d\n",
                bottom_index, n, c, ph, pw, index);
              exit(0);
            }
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
#endif	

    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop 
#ifdef USE_SWPOOL
  if(pooling_judge_condition(bottom[0]->num(),channels_,pooled_height_, pooled_width_) >0)				
	{
		if(sizeof(Dtype) == sizeof(double))
		    pooling_backward_avg_d(bottom[0]->num(),channels_,(const double*)top_diff,(double*)bottom_diff,bottom[0]->offset(0, 1),
				top[0]->offset(0, 1),pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
		else
			pooling_backward_avg_f(bottom[0]->num(),channels_,(const float*)top_diff,(float*)bottom_diff,bottom[0]->offset(0, 1),
				top[0]->offset(0, 1),pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
	}
	else
	{
		for (int n = 0; n < top[0]->num(); ++n) {
		  for (int c = 0; c < channels_; ++c) {
			for (int ph = 0; ph < pooled_height_; ++ph) {
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				int hstart = ph * stride_h_ - pad_h_;
				int wstart = pw * stride_w_ - pad_w_;
				int hend = min(hstart + kernel_h_, height_ + pad_h_);
				int wend = min(wstart + kernel_w_, width_ + pad_w_);
				int pool_size = (hend - hstart) * (wend - wstart);
				hstart = max(hstart, 0);
				wstart = max(wstart, 0);
				hend = min(hend, height_);
				wend = min(wend, width_);
				for (int h = hstart; h < hend; ++h) {
				  for (int w = wstart; w < wend; ++w) {
					bottom_diff[h * width_ + w] +=
					  top_diff[ph * pooled_width_ + pw] / pool_size;
				  }
				}
			  }
			}
			// offset
			bottom_diff += bottom[0]->offset(0, 1);
			top_diff += top[0]->offset(0, 1);
		  }
		}
	}    
#else
	for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
#endif	

    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
