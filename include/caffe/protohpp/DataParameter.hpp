#ifndef _DATAPARAMETER_
#define _DATAPARAMETER_ 
#include <string>

namespace caffe {

class DataParameter {
public:
  enum DB {
    LEVELDB = 0,
    LMDB = 1
  };

  DataParameter(): rand_skip_(0),
    backend_(LMDB),
    scale_(1),
    crop_size_(0),
    mirror_(false),
    force_encoded_color_(false),
    prefetch_(4){}

  DataParameter(const DataParameter& other){
      this->CopyFrom(other);
    }

    inline DataParameter& operator=(const DataParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

  inline void CopyFrom(const DataParameter& other) {
      batch_size_ = other.batch_size();
      data_source_ = other.data_source();
      label_source_ = other.label_source();
      rand_skip_ = other.rand_skip();
      backend_= other.backend();
      scale_ = other.scale();
      crop_size_ = other.crop_size();
      mirror_ = other.mirror();
      force_encoded_color_ = other.force_encoded_color();
      prefetch_ = other.prefetch();
      mean_value_.resize(other.mean_value_size());
      for( int i = 0; i < other.mean_value_size(); ++i )
        mean_value_[i] = other.mean_value(i);
  }
  void set_source(std::string dsource, std::string lsource) { data_source_ = dsource; label_source_ = lsource; }
  inline int batch_size() const { return batch_size_; }
  inline void set_batch_size( int value ) { batch_size_ = value; }
  inline const std::string data_source() const { return data_source_; }
  inline const std::string label_source() const { return label_source_; }
  inline int rand_skip() const { return rand_skip_; }
  inline DB backend() const { return backend_; }
  inline int scale() const { return scale_; }
  inline int crop_size() const { return crop_size_; }
  inline bool mirror() const { return mirror_; } 
  inline bool force_encoded_color() const { return force_encoded_color_; }
  inline int prefetch() const { return prefetch_; }
  inline float mean_value( int idx ) const { return mean_value_[idx]; }
  inline int mean_value_size() const { return mean_value_.size(); }

  private:
    int batch_size_;
    std::string data_source_, label_source_;
    int rand_skip_;
    DB backend_;
    float scale_;
    int crop_size_;
    bool mirror_;
    bool force_encoded_color_;
    int prefetch_;
    std::string mean_file_;
    std::vector<float> mean_value_;
};
}

#endif
