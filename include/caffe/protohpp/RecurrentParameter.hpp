#ifndef _RECURRENTPARAMETER_
#define _RECURRENTPARAMETER_

#include "FillerParameter.hpp"

namespace caffe {

class RecurrentParameter {
  public:
    RecurrentParameter() {
      num_output_ = 1;
      debug_info_ = false;
      expose_hidden_ = false;
    }
    
    RecurrentParameter(const RecurrentParameter& other){
      this->CopyFrom(other);
    }

    inline RecurrentParameter& operator=(const RecurrentParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const RecurrentParameter& other ) {
      num_output_ = other.num_output();
      debug_info_ = other.debug_info();
      expose_hidden_ = other.expose_hidden();
      weight_filler_.CopyFrom(other.weight_filler());
      bias_filler_.CopyFrom(other.bias_filler());
    }

    inline int num_output() const { return num_output_; }
    inline bool debug_info() const { return debug_info_; }
    inline bool expose_hidden() const { return expose_hidden_; }

    inline void set_num_output(int x) { num_output_ = x; }
    inline void set_debug_info(bool x) {debug_info_=x;}
    inline void set_expose_hidden(bool x) {expose_hidden_=x;}

    inline const FillerParameter& bias_filler() const { return bias_filler_; }
    inline const FillerParameter& weight_filler() const { return weight_filler_; }
    inline FillerParameter* mutable_bias_filler() { return &bias_filler_; }
    inline FillerParameter* mutable_weight_filler() { return &weight_filler_; }
    
  private:
    int num_output_;
    bool debug_info_;
    bool expose_hidden_;

    FillerParameter weight_filler_;
    FillerParameter bias_filler_;
};

}

#endif
