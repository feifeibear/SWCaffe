#ifndef _CONVOLUTIONPARAMETER_
#define _CONVOLUTIONPARAMETER_

#include "FillerParameter.hpp"

namespace caffe {


class ConvolutionParameter {
  public:
    ConvolutionParameter() {}
    ConvolutionParameter(int num_output, bool bias_term=true,
                         int dilation=1, int group=1, int axis=1, 
                         bool force_nd_im2col=false):
      num_output_(num_output), bias_term_(bias_term),
      dilation_(dilation), group_(group), axis_(axis), 
      force_nd_im2col_(force_nd_im2col)
    {}

    /** Calling these set_functions are required !
        But should only call either one of set_xxx & set_xxx_2d !
        Normally calling 2d_functions should be enough.
    */
    void set_pad(std::vector<int> pad) {
      pad_ = pad;
      has_pad_h_ = has_pad_w_ = false;
    }
    void set_pad_2d(int pad_h=0, int pad_w=0) {
      pad_h_ = pad_h;
      pad_w_ = pad_w;
      has_pad_h_ = has_pad_w_ = true;
    }
    void set_kernel(std::vector<int> kernel_size) {
      kernel_size_ = kernel_size;
      has_kernel_h_ = has_kernel_w_ = false;
    }
    void set_kernel_2d(int kernel_h, int kernel_w) {
      kernel_h_ = kernel_h;
      kernel_w_ = kernel_w;
      has_kernel_h_ = has_kernel_w_ = true;
    }
    void set_stride(std::vector<int> stride) {
      stride_ = stride;
      has_stride_h_ = has_stride_w_ = false;
    }
    void set_stride_2d(int stride_h=1, int stride_w=1) {
      stride_h_ = stride_h;
      stride_w_ = stride_w;
      has_stride_h_ = has_stride_w_ = true;
    }

    void CopyFrom( const ConvolutionParameter& other ) {
      num_output_ = other.num_output();
      axis_ = other.axis();
      bias_term_ = other.bias_term();
      pad_ = other.pad();
      pad_h_ = other.pad_h();
      pad_w_ = other.pad_w();
      kernel_size_ = other.kernel_size();
      kernel_h_ = other.kernel_h();
      kernel_w_ = other.kernel_w();
      stride_ = other.stride();
      stride_h_ = other.stride_h();
      stride_w_ = other.stride_w();
      weight_filler_= other.weight_filler();
      bias_filler_ = other.bias_filler();
      dilation_ = other.dilation();
      group_ = other.group();
      force_nd_im2col_ = other.force_nd_im2col();
      has_pad_h_ = other.has_pad_h();
      has_pad_w_ = other.has_pad_w();
      has_kernel_h_ = other.has_kernel_h();
      has_kernel_w_ = other.has_kernel_w();
      has_stride_h_ = other.has_stride_h();
      has_stride_w_ = other.has_stride_w();
    }

    inline int axis() const { return axis_; }
    inline int num_output() const { return num_output_; }
    inline const FillerParameter bias_filler() const { return bias_filler_; }
    inline const FillerParameter weight_filler() const { return weight_filler_; }
    inline bool bias_term() const { return bias_term_; }
    inline std::vector<int> pad() const { return pad_; }
    inline int pad_h() const { return pad_h_; }
    inline int pad_w() const { return pad_w_; }
    inline std::vector<int> kernel_size() const { return kernel_size_; }
    inline int kernel_h() const { return kernel_h_; }
    inline int kernel_w() const { return kernel_w_; }
    inline std::vector<int> stride() const { return stride_; }
    inline int stride_h() const { return stride_h_; }
    inline int stride_w() const { return stride_w_; }
    inline int pad_size() const { return pad_.size(); }
    inline int kernel_size_size() const { return kernel_size_.size(); }
    inline int stride_size() const { return stride_.size(); }
    inline std::vector<int> dilation() const { return dilation_; }
    inline int dilation_size() const { return dilation_.size(); }
    inline int group() const { return group_; }
    inline bool force_nd_im2col() const { return force_nd_im2col_; }

    inline bool has_pad_h() const { return has_pad_h_; }
    inline bool has_pad_w() const { return has_pad_w_; }
    inline bool has_kernel_h() const { return has_kernel_h_; }
    inline bool has_kernel_w() const { return has_kernel_w_; }
    inline bool has_stride_h() const { return has_stride_h_; }
    inline bool has_stride_w() const { return has_stride_w_; }
  
  private:
    int num_output_;
    bool bias_term_;
    std::vector<int> pad_;
    std::vector<int> kernel_size_;
    std::vector<int> stride_;

    std::vector<int> dilation_;
    int pad_h_, pad_w_;
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int group_;
    
    FillerParameter weight_filler_;
    FillerParameter bias_filler_;

    int axis_;
    bool force_nd_im2col_;

    bool has_pad_h_, has_pad_w_;
    bool has_kernel_h_, has_kernel_w_;
    bool has_stride_h_, has_stride_w_;
};

}
#endif
