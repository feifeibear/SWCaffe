#ifndef _POOLINGPARAMETER_
#define _POOLINGPARAMETER_

namespace caffe {

enum PoolMethod {
    PoolingParameter_PoolMethod_MAX = 0,
    PoolingParameter_PoolMethod_AVE = 1,
    PoolingParameter_PoolMethod_STOCHASTIC = 2
};

enum PoolingParameter_Engine {
  PoolingParameter_Engine_DEFAULT = 0,
  PoolingParameter_Engine_CAFFE = 1,
  PoolingParameter_Engine_CUDNN = 2
};

class PoolingParameter {
  public:
    PoolingParameter(PoolMethod pool=PoolingParameter_PoolMethod_MAX, bool global_pooling=false,
                     PoolingParameter_Engine engine=PoolingParameter_Engine_CAFFE):
      pool_(pool), global_pooling_(global_pooling),
      engine_(engine)
    {}
    
    /** Calling these set_functions are required !
        But should only call either one of set_xxx & set_xxx_2d !
        Normally calling 2d_functions should be enough.
    */
    void set_pad(int pad=0) {
      pad_ = pad;
      has_pad_ = true;
      has_pad_h_ = has_pad_w_ = false;
    }
    void set_pad_2d(int pad_h=0, int pad_w=0) {
      pad_h_ = pad_h;
      pad_w_ = pad_w;
      has_pad_ = false;
      has_pad_h_ = has_pad_w_ = true;
    }
    void set_kernel(int kernel_size) {
      kernel_size_ = kernel_size;
      has_kernel_size_ = true;
      has_kernel_h_ = has_kernel_w_ = false;
    }
    void set_kernel_2d(int kernel_h, int kernel_w)
    {
      kernel_h_ = kernel_h;
      kernel_w_ = kernel_w;
      has_kernel_size_ = false;
      has_kernel_h_ = has_kernel_w_ = true;
    }
    void set_stride(int stride) {
      stride_ = stride;
      has_stride_ = true;
      has_stride_h_ = has_stride_w_ = false;
    }
    void set_stride_2d(int stride_h, int stride_w) {
      stride_h_ = stride_h;
      stride_w_ = stride_w;
      has_stride_ = false;
      has_stride_h_ = has_stride_w_ = true;
    }

    void CopyFrom( const PoolingParameter& other ) {
      pool_ = other.pool();
      pad_ = other.pad();
      pad_h_ = other.pad_h();
      pad_w_ = other.pad_w();
      kernel_size_ = other.kernel_size();
      kernel_h_ = other.kernel_h();
      kernel_w_ = other.kernel_w();
      stride_ = other.stride();
      stride_h_ = other.stride_h();
      stride_w_ = other.stride_w();
      global_pooling_ = other.global_pooling();
      has_pad_ = other.has_pad();
      has_pad_h_ = other.has_pad_h();
      has_pad_w_ = other.has_pad_w();
      has_kernel_size_ = other.has_kernel_size();
      has_kernel_h_ = other.has_kernel_h();
      has_kernel_w_ = other.has_kernel_w();
      has_stride_ = other.has_stride();
      has_stride_h_ = other.has_stride_h();
      has_stride_w_ = other.has_stride_w();
      engine_ = other.engine();
    }

    inline const PoolMethod pool() const { return pool_; }
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
    inline PoolingParameter_Engine engine() const { return engine_; }

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
    PoolingParameter_Engine engine_;
};

}
#endif
