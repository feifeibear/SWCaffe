#ifndef _DROPOUTPARAMETER_
#define _DROPOUTPARAMETER_

namespace caffe {

class DropoutParameter {
  public:
    DropoutParameter() {
      dropout_ratio_ = 0.5;
    }
    
    DropoutParameter(const DropoutParameter& other){
      this->CopyFrom(other);
    }

    inline DropoutParameter& operator=(const DropoutParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const DropoutParameter& other ) {
      dropout_ratio_ = other.dropout_ratio();
    }

    inline float dropout_ratio() const { return dropout_ratio_; }

    inline void set_dropout_ratio(float x) {dropout_ratio_=x;}
    
  private:
    float dropout_ratio_;
};

}

#endif
