#ifndef _POOLINGPARAMETER_
#define _POOLINGPARAMETER_

namespace caffe {

enum PoolMethod {
    MAX = 0;
    AVE = 1;
    STOCHASTIC = 2;
}

class PoolingParameter {
  public:
    PoolingParameter() {}
    PoolingParameter(int pad=0, int pad_h=0, int pad_w=0,
                     int kernel_size=2, int kernel_h=2, int kernel_w=2,
                     int stride=1, int stride_h=1, int stride_w=1
                     PoolMethod pool=MAX):
      pad_(pad), pad_h_(pad_h), pad_w_(pad_w),
      kernel_size_(kernel_size), kernel_h_(kernel_h), kernel_w_(kernel_w),
      stride_(stride), stride_h_(stride_h), stride_w_(stride_w),
      pool_(pool)
    {}
    
    void CopyFrom( const InnerProductParameter& other ) {
      pool_ = other.pool();
      pad_ = other.pad();
      pad_h_ = other.pad_h();
      pad_w_ = other.pad_w();
      kernel_ = other.kernel();
      kernel_h_ = other.kernel_h();
      kernel_w_ = other.kernel_w();
      stride_ = other.stride();
      stride_h_ = other.stride_h();
      stride_w_ = other.stride_w();
    }

    inline const PoolMethod& pool() const { return pool_; }
    inline int pad() const { return pad_; }
    inline int pad_h() const { return pad_h_; }
    inline int pad_w() const { return pad_w_; }
    inline int kernel_size() const { return kernel_size_; }
    inline int kernel_h() const { return kernel_h_; }
    inline int kernel_w() const { return kernel_w_; }
    inline int stride() const { return stride_; }
    inline int stride_h() const { return stride_h_; }
    inline int stride_w() const { return stride_w_; }
    inline bool global_pooling() const { return global_pooling_; }

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
    PoolMethod pool_;
    int pad_, pad_h_, pad_w_;
    int kernel_size_, kernel_h_, kernel_w_;
    int stride_, stride_h_, stride_w_;
    bool global_pooling_;
    bool has_pad_, has_pad_h_, has_pad_w_;
    bool has_kernel_size_, has_kernel_h_, has_kernel_w_;
    bool has_stride_, has_stride_h_, has_stride_w_;
};

}
#endif
