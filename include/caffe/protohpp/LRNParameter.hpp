#ifndef _LRNPARAMETER_HPP_
#define _LRNPARAMETER_HPP_

namespace caffe {

enum LRNParameter_Engine {
    LRNParameter_Engine_DEFAULT = 0,
    LRNParameter_Engine_CAFFE = 1,
    LRNParameter_Engine_CUDNN = 2
  };
enum LRNParameter_NormRegion {
    LRNParameter_NormRegion_ACROSS_CHANNELS = 0,
    LRNParameter_NormRegion_WITHIN_CHANNEL = 1
  };

class LRNParameter {
  public:
    LRNParameter():local_size_(1),alpha_(1),beta_(0.75),
      norm_region_(LRNParameter_NormRegion_ACROSS_CHANNELS), k_(5), 
      engine_(LRNParameter_Engine_DEFAULT) {}
    ~LRNParameter() {}
    inline int local_size() const { return local_size_; }
    inline float alpha() const { return alpha_; }
    inline float beta() const { return beta_; }
    inline LRNParameter_NormRegion norm_region() const { return norm_region_; }
    inline float k() const { return k_; } 
    inline LRNParameter_Engine engine() const { return engine_; }

    inline void set_local_size( int n ) { local_size_ = n; }
    inline void set_alpha( float n ) { alpha_ = n; }
    inline void set_beta( float n ) { beta_ = n; }
    inline void set_norm_region( LRNParameter_NormRegion n ) { norm_region_ = n; }
    inline void set_k( float n ) { k_ = n; }
    inline void set_engine( LRNParameter_Engine n ) { engine_ = n; }

    void CopyFrom( const LRNParameter& other ) {
      local_size_ = other.local_size();
      alpha_ = other.alpha();
      beta_ = other.beta();
      norm_region_ = other.norm_region();
      k_ = other.k();
      engine_ = other.engine();
    }
    LRNParameter(const LRNParameter& other) { this->CopyFrom(other); }

    inline LRNParameter& operator=(const LRNParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

  private:
    int local_size_;
    float alpha_;
    float beta_;
    LRNParameter_NormRegion norm_region_;
    float k_;
    LRNParameter_Engine engine_;
};

}
#endif
