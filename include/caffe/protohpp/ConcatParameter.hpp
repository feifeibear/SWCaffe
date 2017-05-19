#ifndef _CONCATPARAMETER_
#define _CONCATPARAMETER_

namespace caffe {

class ConcatParameter {
  public:
    ConcatParameter() {
      axis_ = 1;
      concat_dim_ = 1;
      has_axis_ = has_concat_dim_ = false;
    }
    
    ConcatParameter(const ConcatParameter& other){
      this->CopyFrom(other);
    }

    inline ConcatParameter& operator=(const ConcatParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const ConcatParameter& other ) {
      axis_ = other.axis();
      concat_dim_ = other.concat_dim();
      has_axis_ = other.has_axis();
      has_concat_dim_ = other.has_concat_dim();
    }

    inline int axis() const { return axis_; }
    inline int concat_dim() const { return concat_dim_; }
    inline bool has_axis() const { return has_axis_; }
    inline bool has_concat_dim() const { return has_concat_dim_; }

    inline void set_axis(int x) { has_axis_ = true; axis_ = x; }
    inline void set_concat_dim(int x) { has_concat_dim_ = true; concat_dim_=x;}

    static ConcatParameter default_instance_;
    
  private:
    int axis_;
    int concat_dim_;

    bool has_axis_, has_concat_dim_;
};

}

#endif
