#ifndef _INNERPRODUCTPARAMETER_
#define _INNERPRODUCTPARAMETER_
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
    value_(0.0),
    min_(0.0),
    max_ (1.0),
    mean_ (0.0),
    std_ (0.0),
    sparse_(1),
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

class InnerProductParameter {
  public:
    InnerProductParameter() {}
    explicit InnerProductParameter(int num_output):num_output_(num_output),
      bias_term_(true),
      transpose_(false),
      axis_(1)
    {}
    void CopyFrom( const InnerProductParameter& other ) {
      num_output_ = other.num_output();
      axis_ = other.axis();
      bias_term_ = other.bias_term();
      transpose_ = other.transpose();
      weight_filler_= other.weight_filler();
      bias_filler_ = other.bias_filler();
    }
    inline int axis() const { return axis_; }
    inline int num_output() const { return num_output_; }
    inline const FillerParameter& bias_filler() const { return bias_filler_; }
    inline const FillerParameter& weight_filler() const { return weight_filler_; }
    inline bool bias_term() const { return bias_term_; }
    inline bool transpose() const { return transpose_; }
  private:
    int num_output_;
    int axis_;
    bool bias_term_;
    bool transpose_;

    FillerParameter weight_filler_;
    FillerParameter bias_filler_;
};

}
#endif
