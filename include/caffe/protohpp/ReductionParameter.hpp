#ifndef _REDUCTIONPARAMETER_
#define _REDUCTIONPARAMETER_

#include <vector>

namespace caffe {

enum ReductionParameter_ReductionOp {
  ReductionParameter_ReductionOp_SUM = 1,
  ReductionParameter_ReductionOp_ASUM = 2,
  ReductionParameter_ReductionOp_SUMSQ = 3,
  ReductionParameter_ReductionOp_MEAN = 4
};

class ReductionParameter {
  public:
    ReductionParameter() {
      axis_ = 0;
      has_axis_ = false;
      coeff_ = 1;
      operation_ = ReductionParameter_ReductionOp_SUM;
    }
    
    ReductionParameter(const ReductionParameter& other){
      this->CopyFrom(other);
    }

    inline ReductionParameter& operator=(const ReductionParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const ReductionParameter& other ) {
      axis_ = other.axis();
      operation_ = other.operation();
      coeff_ = other.coeff();
      has_axis_ = other.has_axis();
    }

    inline int axis() const { return axis_; }
    inline ReductionParameter_ReductionOp operation() const { return operation_; }
    inline float coeff() const { return coeff_; }
    inline bool has_axis() const { return has_axis_; }

    inline void set_axis(int x) { has_axis_ = true; axis_ = x; }
    inline void set_operation(ReductionParameter_ReductionOp x) { operation_ = x; }
    inline void set_coeff(float x) { coeff_ = x; }

    static ReductionParameter default_instance_;
    
  private:
    int axis_;
    ReductionParameter_ReductionOp operation_;
    float coeff_;

    bool has_axis_;
};

}

#endif
