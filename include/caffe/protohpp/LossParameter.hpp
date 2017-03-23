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
      has_ignore_label_ = has_normalize_=  has_normalization_ = false;
    }

    LossParameter(const LossParameter& other){
      this->CopyFrom(other);
    }

    inline LossParameter& operator=(const LossParameter& other) {
      this->CopyFrom(other);
      return *this;
    }
    
    inline void CopyFrom( const LossParameter& other ) {
      ignore_label_ = other.ignore_label();
      normalization_ = other.normalization();
      normalize_ = other.normalize();
      has_ignore_label_ = other.has_ignore_label();
      has_normalization_ = other.has_normalization();
      has_normalize_ = other.has_normalize();
    }

    inline int ignore_label() const { return ignore_label_; }
    inline LossParameter_NormalizationMode normalization() const { return normalization_; }
    inline bool normalize() const { return normalize_; }
    inline bool has_normalize() const { return has_normalize_; }
    inline bool has_normalization() const { return has_normalization_; }
    inline bool has_ignore_label() const { return has_ignore_label_; }

    inline void set_ignore_label(int x) { ignore_label_ = x; has_ignore_label_ = true; }
    inline void set_normalization(LossParameter_NormalizationMode x) { normalization_=x; has_normalization_ = true; }
    inline void set_normalize(bool x) { normalize_=x; has_normalize_ = true; }

    static LossParameter default_instance_;

  private:
    int ignore_label_;
    LossParameter_NormalizationMode normalization_;
    bool normalize_;

    bool has_ignore_label_, has_normalization_, has_normalize_;
};

}

#endif
