#ifndef _ACCURACY_PARAMETER_H_
#define _ACCURACY_PARAMETER_H_

namespace caffe {

class AccuracyParameter {
  public:
    AccuracyParameter():top_k_(1), axis_(1) {
      has_ignore_label_ = false;
    }
    
    inline void CopyFrom( const AccuracyParameter& other ) {
      top_k_ = other.top_k();
      axis_ = other.axis();
      ignore_label_ = other.ignore_label();
      has_ignore_label_ = other.has_ignore_label();
    }

    inline int ignore_label() const { return ignore_label_; }
    inline int top_k() const { return top_k_; }
    inline int axis() const { return axis_; }
    inline bool has_ignore_label() const { return has_ignore_label_; }

  private:
    int top_k_;
    int axis_;
    int ignore_label_;

    bool has_ignore_label_;
};

}

#endif
