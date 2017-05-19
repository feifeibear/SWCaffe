#ifndef _FILLERPARAMETER_
#define _FILLERPARAMETER_
#include <string>

namespace caffe {


enum FillerParameter_VarianceNorm {
  FillerParameter_VarianceNorm_FAN_IN = 0,
  FillerParameter_VarianceNorm_FAN_OUT = 1,
  FillerParameter_VarianceNorm_AVERAGE = 2
};

class FillerParameter {
public:
  FillerParameter():
    type_("constant"), value_(0.0),
    min_(0.0), max_ (1.0), mean_ (0.0),
    std_ (1.0), sparse_(-1.0), variance_norm_(FillerParameter_VarianceNorm_FAN_IN) {}

  FillerParameter(const FillerParameter& other) {
    this->CopyFrom(other);
  }

  FillerParameter& operator=(const FillerParameter& other){
    this->CopyFrom(other);
    return *this;
  }

  inline std::string type() const { return type_; }
  inline float value() const { return value_; }
  inline float min() const { return min_; }
  inline float mean() const { return mean_; }
  inline float std() const { return std_; }
  inline float sparse() const { return sparse_; }
  inline float max() const { return max_; }
  FillerParameter_VarianceNorm variance_norm() const { return variance_norm_; }

  void set_type( const std::string type ) { type_ = type; }
  void set_value( const float value ) { value_ = value; }
  void set_min( const float min ) { min_ = min; }
  void set_mean( const float mean ) { mean_ = mean; }
  void set_std( const float std ) { std_ = std; }
  void set_sparse( const float sparse ) { sparse_ = sparse; }
  void set_max( const float max ) { max_ = max; }
  void set_variance_norm( const FillerParameter_VarianceNorm variance_norm ) { variance_norm_ = variance_norm; }

  void CopyFrom(const FillerParameter& other) {
    type_ = other.type();
    value_ = other.value();
    min_ = other.min();
    max_ = other.max();
    mean_ = other.mean();
    std_ = other.std();
    sparse_ = other.sparse();
    variance_norm_ = other.variance_norm();
  }

private:
  std::string type_;
  float value_;
  float min_;
  float max_;
  float mean_;
  float std_;
  int sparse_;
  FillerParameter_VarianceNorm variance_norm_;
};

}
#endif
