#ifndef _SOFTMAX_PARAMETER_H_
#define _SOFTMAX_PARAMETER_H_

namespace caffe {
enum SoftmaxParameter_Engine {
  SoftmaxParameter_Engine_DEFAULT = 0,
  SoftmaxParameter_Engine_CAFFE = 1,
  SoftmaxParameter_Engine_CUDNN = 2
};

class SoftmaxParameter {
  public:
    SoftmaxParameter():engine_(SoftmaxParameter_Engine_DEFAULT),
      axis_(1) {}

    SoftmaxParameter(const SoftmaxParameter& other){
      this->CopyFrom(other);
    }

    inline SoftmaxParameter& operator=(const SoftmaxParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    inline void CopyFrom( const SoftmaxParameter& other ) {
      engine_ = other.engine();
      axis_ = other.axis();
    }
    inline SoftmaxParameter_Engine engine() const { return engine_; }
    inline int axis() const { return axis_; }

    inline void set_engine(SoftmaxParameter_Engine e) { engine_ = e; }
    inline void set_axis(int a) {axis_=a;}

    static SoftmaxParameter default_instance_;

  private:
    SoftmaxParameter_Engine engine_;
    int axis_;

};

}

#endif
