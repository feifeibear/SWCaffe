#ifndef _ELTWISEPARAMETER_
#define _ELTWISEPARAMETER_

namespace caffe {

enum EltwiseParameter_EltwiseOp {
  EltwiseParameter_EltwiseOp_PROD = 0,
  EltwiseParameter_EltwiseOp_SUM = 1,
  EltwiseParameter_EltwiseOp_MAX = 2
};

class EltwiseParameter {
  public:
    EltwiseParameter() {
      operation_ = EltwiseParameter_EltwiseOp_SUM;
      stable_prod_grad_ = true;
    }
    
    EltwiseParameter(const EltwiseParameter& other){
      this->CopyFrom(other);
    }

    inline EltwiseParameter& operator=(const EltwiseParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const EltwiseParameter& other ) {
      operation_ = other.operation();
      coeff_ = other.coeff();
      stable_prod_grad_ = other.stable_prod_grad();
    }

    inline EltwiseParameter_EltwiseOp operation() const { return operation_; }
    inline std::vector<float> coeff() const { return coeff_; }
    inline float coeff(int i) const { return coeff_[i]; }
    inline int coeff_size() const { return coeff_.size(); }
    inline bool stable_prod_grad() const { return stable_prod_grad_; }

    inline void set_operation(EltwiseParameter_EltwiseOp x) { operation_ = x; }
    inline void add_coeff(float x) {coeff_.push_back(x);}
    inline void set_stable_prod_grad(bool x) {stable_prod_grad_=x;}
    
  private:
    EltwiseParameter_EltwiseOp operation_;
    std::vector<float> coeff_;
    bool stable_prod_grad_;
};

}

#endif
