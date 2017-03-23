#ifndef _CONVOLUTIONPARAMETER_
#define _CONVOLUTIONPARAMETER_

#include "FillerParameter.hpp"

namespace caffe {

enum ConvolutionParameter_Engine {
  ConvolutionParameter_Engine_DEFAULT = 0,
  ConvolutionParameter_Engine_CAFFE = 1,
  ConvolutionParameter_Engine_CUDNN = 2
};

class ConvolutionParameter {
  public:
    ConvolutionParameter() {
      num_output_ = 1;
      bias_term_ = true;
      pad_h_ = pad_w_ = 0;
      axis_ = 1;
      stride_h_ = stride_w_ = 1;
      group_ = 1;
      force_nd_im2col_ = false;
      engine_ = ConvolutionParameter_Engine_DEFAULT;
      has_pad_h_ = has_pad_w_ = has_stride_h_ = has_stride_w_ = false;
      has_kernel_h_ = has_kernel_w_ = false;
    }

    ConvolutionParameter(const ConvolutionParameter& other){
      this->CopyFrom(other);
    }

    inline ConvolutionParameter& operator=(const ConvolutionParameter& other) {
      this->CopyFrom(other);
      return *this;
    }
    
    inline int axis() const { return axis_; }
    inline int num_output() const { return num_output_; }
    inline bool bias_term() const { return bias_term_; }
    inline int pad_h() const { return pad_h_; }
    inline int pad_w() const { return pad_w_; }
    inline int kernel_h() const { return kernel_h_; }
    inline int kernel_w() const { return kernel_w_; }
    inline int stride_h() const { return stride_h_; }
    inline int stride_w() const { return stride_w_; }
    inline int group() const { return group_; }
    inline bool force_nd_im2col() const { return force_nd_im2col_; }
    inline ConvolutionParameter_Engine engine() const { return engine_; }

    inline bool has_pad_h() const { return has_pad_h_; }
    inline bool has_pad_w() const { return has_pad_w_; }
    inline bool has_kernel_h() const { return has_kernel_h_; }
    inline bool has_kernel_w() const { return has_kernel_w_; }
    inline bool has_stride_h() const { return has_stride_h_; }
    inline bool has_stride_w() const { return has_stride_w_; }

    inline void set_axis(int a) { axis_ = a; }
    inline void set_num_output(int n) { num_output_ = n; }
    inline void set_bias_term(bool b) { bias_term_ = b; }
    inline void set_pad_h(int p) { pad_h_ = p; has_pad_h_ = true; }
    inline void set_pad_w(int p) { pad_w_ = p; has_pad_w_ = true;}
    inline void set_kernel_h(int k) { kernel_h_ = k; has_kernel_h_ = true;}
    inline void set_kernel_w(int k) { kernel_w_ = k; has_kernel_w_ = true;}
    inline void set_stride_h(int s) { stride_h_ = s; has_stride_h_ = true;}
    inline void set_stride_w(int s) { stride_w_ = s; has_stride_w_ = true;}
    inline void set_group(int g) { group_ = g; }
    inline void set_force_nd_im2col(bool f) { force_nd_im2col_ = f; }
    inline void set_engine(ConvolutionParameter_Engine e) { engine_ = e;}

    inline std::vector<int> pad() const { return pad_; }
    inline int pad(int i) const { return pad_[i]; }
    inline int pad_size() const { return pad_.size(); }
    inline void add_pad(int p) { pad_.push_back(p); }

    inline std::vector<int> kernel_size() const { return kernel_size_; }
    inline int kernel_size(int i) const { return kernel_size_[i]; }
    inline int kernel_size_size() const { return kernel_size_.size(); }
    inline void add_kernel_size(int k) { kernel_size_.push_back(k); }
    inline void clear_kernel_size() { kernel_size_.clear(); }

    inline std::vector<int> stride() const { return stride_; }
    inline int stride(int i) const { return stride_[i]; }
    inline int stride_size() const { return stride_.size(); }
    inline void add_stride(int s) { stride_.push_back(s); }
    inline void clear_stride() { stride_.clear(); }

    inline std::vector<int> dilation() const { return dilation_; }
    inline int dilation(int i) const { return dilation_[i]; }
    inline int dilation_size() const { return dilation_.size(); }
    inline void add_dilation(int d) { dilation_.push_back(d); }

    inline const FillerParameter& bias_filler() const { return bias_filler_; }
    inline FillerParameter* mutable_bias_filler() { return &bias_filler_; }
    inline const FillerParameter& weight_filler() const { return weight_filler_; }
    inline FillerParameter* mutable_weight_filler() { return &weight_filler_; }

    void CopyFrom( const ConvolutionParameter& other ) {
      num_output_ = other.num_output();
      axis_ = other.axis();
      bias_term_ = other.bias_term();
      pad_ = other.pad();
      pad_h_ = other.pad_h();
      pad_w_ = other.pad_w();
      kernel_size_ = other.kernel_size();
      kernel_h_ = other.kernel_h();
      kernel_w_ = other.kernel_w();;

      stride_ = other.stride();
      stride_h_ = other.stride_h();
      stride_w_ = other.stride_w();
      weight_filler_.CopyFrom(other.weight_filler());
      bias_filler_.CopyFrom(other.bias_filler());
      dilation_ = other.dilation();
      group_ = other.group();
      force_nd_im2col_ = other.force_nd_im2col();
      has_pad_h_ = other.has_pad_h();
      has_pad_w_ = other.has_pad_w();
      has_kernel_h_ = other.has_kernel_h();
      has_kernel_w_ = other.has_kernel_w();
      has_stride_h_ = other.has_stride_h();
      has_stride_w_ = other.has_stride_w();
      engine_ = other.engine();
    }
  
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

    ConvolutionParameter_Engine engine_;
};

}
#endif
