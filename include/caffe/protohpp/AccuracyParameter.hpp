#ifndef _ACCURACY_PARAMETER_H_
#define _ACCURACY_PARAMETER_H_

namespace caffe {

class AccuracyParameter {
  public:
    AccuracyParameter():top_k_(1), axis_(1) {
      has_ignore_label_ = false;
    }

    AccuracyParameter(const AccuracyParameter& other){
      this->CopyFrom(other);
    }

    inline AccuracyParameter& operator=(const AccuracyParameter& other) {
      this->CopyFrom(other);
      return *this;
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

    inline void set_ignore_label(int x) { ignore_label_=x; has_ignore_label_ = true; }
    inline void set_top_k(int x) { top_k_=x; }
    inline void set_axis(int x) { axis_=x; }

    static AccuracyParameter default_instance_;

  private:
    int top_k_;
    int axis_;
    int ignore_label_;

    bool has_ignore_label_;
};

}

#endif
