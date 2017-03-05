#ifndef _CONVOLUTIONPARAMETER_
#define _CONVOLUTIONPARAMETER_

namespace caffe {


class ConvolutionParameter {
  public:
    ConvolutionParameter() {}
    ConvolutionParameter(int num_output, bool bias_term=true,
                         int* pad=NULL, int* kernel_size=NULL, int* stride=NULL,
                         int dilation=1, int pad_h=0, int pad_w=0,
                         int kernel_h=0, int kernel_w=0, int stride_h=0, int stride_w=0,
                         int group=1, int axis=1, bool force_nd_im2col=false):
      num_output_(num_output), bias_term_(bias_term),
      pad_(pad), kernel_size_(kernel_size), stride_(stride),
      dilation_(dilation), pad_h_(pad_h), pad_w_(pad_w),
      kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w),
      group_(group), axis_(axis), force_nd_im2col_(force_nd_im2col)
    {}
    void CopyFrom( const ConvolutionParameter& other ) {
      num_output_ = other.num_output();
      axis_ = other.axis();
      bias_term_ = other.bias_term();
      pad_ = other.pad();
      pad_h_ = other.pad_h();
      pad_w_ = other.pad_w();
      kernel_ = other.kernel();
      kernel_h_ = other.kernel_h();
      kernel_w_ = other.kernel_w();
      stride_ = other.stride();
      stride_h_ = other.stride_h();
      stride_w_ = other.stride_w();
      weight_filler_= other.weight_filler();
      bias_filler_ = other.bias_filler();
      dilation_ = other.dilation();
      group_ = other.group();
      force_nd_im2col_ = other.force_nd_im2col_();
    }
    inline int axis() const { return axis_; }
    inline int num_output() const { return num_output_; }
    inline const FillerParameter& bias_filler() const { return bias_filler_; }
    inline const FillerParameter& weight_filler() const { return weight_filler_; }
    inline bool bias_term() const { return bias_term_; }
    inline int* pad() const { return pad_; }
    inline int pad_h() const { return pad_h_; }
    inline int pad_w() const { return pad_w_; }
    inline int* kernel_size() const { return kernel_size_; }
    inline int kernel_h() const { return kernel_h_; }
    inline int kernel_w() const { return kernel_w_; }
    inline int* stride() const { return stride_; }
    inline int stride_h() const { return stride_h_; }
    inline int stride_w() const { return stride_w_; }

    inline bool has_pad() const { return has_pad_; }
    inline bool has_pad_h() const { return has_pad_h_; }
    inline bool has_pad_w() const { return has_pad_w_; }
    inline bool has_kernel_size() const { return has_kernel_size_; }
    inline bool has_kernel_h() const { return has_kernel_h_; }
    inline bool has_kernel_w() const { return has_kernel_w_; }
    inline bool has_stride() const { return has_stride_; }
    inline bool has_stride_h() const { return has_stride_h_; }
    inline bool has_stride_w() const { return has_stride_w_; }
  
  private:
    int num_output_;
    bool bias_term_;
    int* pad_;
    int* kernel_size_;
    int* stride_;

    int dilation_;
    int pad_h_, pad_w_;
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int group_;
    
    FillerParameter weight_filler_;
    FillerParameter bias_filler_;

    int axis_;
    bool force_nd_im2col_;
};

}
#endif
