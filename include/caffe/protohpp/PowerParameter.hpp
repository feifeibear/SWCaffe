#ifndef _POWERPARAMETER_H_
#define _POWERPARAMETER_H_

namespace caffe {
  class PowerParameter {
    public:
      PowerParameter():power_(1.0), scale_(1.0), shift_(0.0) {}
      ~PowerParameter() {}

      inline PowerParameter& operator=(const PowerParameter& other) {
        this->CopyFrom(other);
        return *this;
      }

      inline void CopyFrom( const PowerParameter& other ) {
        power_ = other.power();
        scale_ = other.scale();
        shift_ = other.shift();
      }

      inline float power() const { return power_; }
      inline float scale() const { return scale_; }
      inline float shift() const { return shift_; }

      inline void set_power(float n) { power_ = n; }
      inline void set_scale(float n) { scale_ = n; }
      inline void set_shift(float n) { shift_ = n; }

    private:
      float power_;
      float scale_;
      float shift_;
  };
}

#endif
