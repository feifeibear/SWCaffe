#ifndef _INNERPRODUCTPARAMETER_
#define _INNERPRODUCTPARAMETER_

#include "FillerParameter.hpp"

namespace caffe {

class InnerProductParameter {
  public:
    InnerProductParameter() {
      num_output_ = 1;
      bias_term_ = true;
      transpose_ = false;
      axis_= 1;
    }
    InnerProductParameter(const InnerProductParameter& other){
      this->CopyFrom(other);
    }

    inline InnerProductParameter& operator=(const InnerProductParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const InnerProductParameter& other ) {
      num_output_ = other.num_output();
      axis_ = other.axis();
      bias_term_ = other.bias_term();
      transpose_ = other.transpose();
      weight_filler_.CopyFrom(other.weight_filler());
      bias_filler_.CopyFrom(other.bias_filler());
    }

    inline int axis() const { return axis_; }
    inline int num_output() const { return num_output_; }
    inline bool bias_term() const { return bias_term_; }
    inline bool transpose() const { return transpose_; }

    inline void set_num_output(int N) { num_output_ = N; }
    inline void set_axis(int x) {axis_=x;}
    inline void set_bias_term(bool x) {bias_term_=x;}
    inline void set_transpose(bool x) {transpose_=x;}

    inline const FillerParameter& bias_filler() const { return bias_filler_; }
    inline const FillerParameter& weight_filler() const { return weight_filler_; }
    inline FillerParameter* mutable_bias_filler() { return &bias_filler_; }
    inline FillerParameter* mutable_weight_filler() { return &weight_filler_; }
  private:
    int num_output_;
    int axis_;
    bool bias_term_;
    bool transpose_;

    FillerParameter weight_filler_;
    FillerParameter bias_filler_;
};

}
#endif
