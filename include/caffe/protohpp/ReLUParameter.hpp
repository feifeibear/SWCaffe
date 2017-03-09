#ifndef _RELUPARAMETER_H_
#define _RELUPARAMETER_H_

namespace caffe {

enum ReLUParameter_Engine{
    ReLUParameter_Engine_DEFAULT= 0,
    ReLUParameter_Engine_CAFFE = 1,
    ReLUParameter_Engine_CUDNN = 2
};

class ReLUParameter {
public:

  ReLUParameter():negative_slope_(0), engine_(ReLUParameter_Engine_DEFAULT){}

  void CopyFrom(const ReLUParameter& other) {
    negative_slope_ = other.negative_slope();
    engine_ = other.engine();
  }

  inline const float negative_slope () const { return negative_slope_; }
  inline const ReLUParameter_Engine engine () const { return engine_; }
  inline void negative_slop ( float value ) { negative_slope_ = value; }
  inline void set_engine (ReLUParameter_Engine value ) { engine_ = value; }
private:
  float negative_slope_;
  ReLUParameter_Engine engine_;
};

}
#endif
