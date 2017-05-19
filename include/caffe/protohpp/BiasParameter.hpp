#ifndef _BIASPARAMETER_
#define _BIASPARAMETER_

#include "FillerParameter.hpp"

namespace caffe {

class BiasParameter {
  public:
    BiasParameter() {
      axis_ = 1;
      num_axes_ = 1;
    }
    
    BiasParameter(const BiasParameter& other){
      this->CopyFrom(other);
    }

    inline BiasParameter& operator=(const BiasParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const BiasParameter& other ) {
      axis_ = other.axis();
      num_axes_ = other.num_axes();
    }

    inline int axis() const { return axis_; }
    inline int num_axes() const { return num_axes_; }

    inline void set_axis(int x) { axis_ = x; }
    inline void set_num_axes(int x) {num_axes_ = x;}
    
    inline const FillerParameter& filler() const { return filler_; }
    inline FillerParameter* mutable_filler() { return &filler_; }

    static BiasParameter default_instance_;

  private:
    int axis_;
    int num_axes_;
    FillerParameter filler_;
};

}

#endif
