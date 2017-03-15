#ifndef _LOSS_PARAMETER_H_
#define _LOSS_PARAMETER_H_

namespace caffe {

enum LossParameter_NormalizationMode {
  LossParameter_NormalizationMode_FULL = 0,
  LossParameter_NormalizationMode_VALID = 1,
  LossParameter_NormalizationMode_BATCH_SIZE = 2,
  LossParameter_NormalizationMode_NONE = 3
};

class LossParameter {
  public:
    LossParameter():normalization_(LossParameter_NormalizationMode_VALID) {
      has_ignore_label_ = false;
    }
    
    inline void CopyFrom( const LossParameter& other ) {
      ignore_label_ = other.ignore_label();
      normalization_ = other.normalization();
      normalize_ = other.normalize();
      has_ignore_label_ = other.has_ignore_label();
    }

    inline int ignore_label() const { return ignore_label_; }
    inline LossParameter_NormalizationMode normalization() const { return normalization_; }
    inline bool normalize() const { return normalize_; }
    inline bool has_normalize() const { return false; }
    inline bool has_normalization() const { return true; }
    inline bool has_ignore_label() const { return has_ignore_label_; }

  private:
    int ignore_label_;
    LossParameter_NormalizationMode normalization_;
    bool normalize_;

    bool has_ignore_label_;
};

}

#endif
