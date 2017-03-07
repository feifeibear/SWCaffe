#ifndef _FILLERPARAMETER_
#define _FILLERPARAMETER_
#include <string>

namespace caffe {


enum VarianceNorm {
    FAN_IN = 0,
    FAN_OUT = 1,
    AVERAGE = 2
  };

class FillerParameter {
  public: 
  FillerParameter(): 
    type_("constant"),
    value_(0.0),
    min_(0.0),
    max_ (1.0),
    mean_ (0.0),
    std_ (0.0),
    sparse_(-1.0),
    variance_norm_(FAN_IN)
    {}
  std::string type() const { return type_; }
  float value() const { return value_; }
  float min() const { return min_; }
  float mean() const { return mean_; }
  float std() const { return std_; }
  float sparse() const { return sparse_; }
  float max() const { return max_; }
  VarianceNorm variance_norm() const { return variance_norm_; }
  void CopyFrom(const FillerParameter& other) {
    value_ = other.value();
    min_ = other.min();
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
  VarianceNorm variance_norm_;
};

}
#endif