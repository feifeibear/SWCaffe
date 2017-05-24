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
    PoolingParameter() {
      has_pad_ = has_pad_h_ = has_pad_w_ = false;
      has_kernel_size_ = has_kernel_h_ = has_kernel_w_ = false;
      has_stride_ = has_stride_h_ = has_stride_w_ = false;
      pad_ = pad_w_ = pad_h_ = 0;
      stride_ = stride_h_ = stride_w_ = 1;
      pool_ = PoolingParameter_PoolMethod_MAX;
      engine_ = PoolingParameter_Engine_DEFAULT;
      global_pooling_ = false;
    }
    
    PoolingParameter(const PoolingParameter& other){
      this->CopyFrom(other);
    }

    inline PoolingParameter& operator=(const PoolingParameter& other) {
      this->CopyFrom(other);
      return *this;
    }
    
    inline PoolMethod pool() const { return pool_; }
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

    inline void set_pool(PoolMethod p) { pool_ = p; }
    inline void set_pad(int p) { pad_ = p; has_pad_ = true; }
    inline void set_pad_h(int p) { pad_h_ = p; has_pad_h_ = true; }
    inline void set_pad_w(int p) { pad_w_ = p; has_pad_w_ = true; }
    inline void set_kernel_size(int k) { kernel_size_ = k; has_kernel_size_ = true;}
    inline void set_kernel_h(int k) { kernel_h_ = k; has_kernel_h_ = true;}
    inline void set_kernel_w(int k) { kernel_w_ = k; has_kernel_w_ = true;}
    inline void set_stride(int s) { stride_ = s; has_stride_ = true;}
    inline void set_stride_h(int s) { stride_h_ = s; has_stride_h_ = true;}
    inline void set_stride_w(int s) { stride_w_ = s; has_stride_w_ = true;}
    inline void set_engine(PoolingParameter_Engine e) { engine_ = e;}
    inline void set_global_pooling(bool g) { global_pooling_ = g; }

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
