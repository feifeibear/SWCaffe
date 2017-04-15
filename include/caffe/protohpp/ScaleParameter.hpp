#ifndef _SCALEPARAMETER_
#define _SCALEPARAMETER_

#include "FillerParameter.hpp"

namespace caffe {

class ScaleParameter {
  public:
    ScaleParameter() {
      axis_ = 1;
      num_axes_ = 1;
      bias_term_ = false;
      has_filler_ = false;
    }
    
    ScaleParameter(const ScaleParameter& other){
      this->CopyFrom(other);
    }

    inline ScaleParameter& operator=(const ScaleParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const ScaleParameter& other ) {
      axis_ = other.axis();
      num_axes_ = other.num_axes();
      bias_term_ = other.bias_term();
      has_filler_ = other.has_filler();
    }

    inline int axis() const { return axis_; }
    inline int num_axes() const { return num_axes_; }
    inline bool bias_term() const { return bias_term_; }
    inline bool has_filler() const { return has_filler_; }

    inline void set_axis(int x) { axis_ = x; }
    inline void set_num_axes(int x) {num_axes_ = x;}
    inline void set_bias_term(bool x) {bias_term_=x;}
    
    inline const FillerParameter& bias_filler() const { return bias_filler_; }
    inline const FillerParameter& filler() const { return filler_; }
    inline FillerParameter* mutable_bias_filler() { return &bias_filler_; }
    inline FillerParameter* mutable_filler() { has_filler_ = true; return &filler_; }

    static ScaleParameter default_instance_;

  private:
    int axis_;
    int num_axes_;
    FillerParameter filler_;
    bool bias_term_;
    FillerParameter bias_filler_;

    bool has_filler_;
};

}

#endif
