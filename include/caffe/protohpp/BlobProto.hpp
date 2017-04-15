#ifndef _PROTOBLOB_H_
#define _PROTOBLOB_H_

class BlobShape {
public:
  BlobShape() {}
  BlobShape(const BlobShape& other){
    this->CopyFrom(other);
  }
  inline void add_dim(int d) { dim_.push_back(d); }
  inline int dim(int i) const { return dim_[i]; }
  inline std::vector<int> dim() const { return dim_; }
  inline int dim_size() const { return dim_.size(); }
  void CopyFrom(const BlobShape& other) {
    dim_ = other.dim();
  }
  void Clear() {
    dim_.clear();
  }
private:
  std::vector<int> dim_;
};

class BlobProto {
  public:
    BlobProto() { has_num_ = has_channels_ = has_height_ = has_width_ = false; }
    BlobProto(const BlobProto& other){
      this->CopyFrom(other);
    }

    bool has_num() const { return has_num_; }
    bool has_channels() const { return has_channels_; }
    bool has_height() const { return has_height_; }
    bool has_width() const { return has_width_; }

    int num() const { return num_; }
    int channels() const { return channels_; }
    int height() const { return height_; }
    int width() const { return width_; }
    const BlobShape& shape() const { return shape_; }

    inline void set_num(int x) { num_ = x; has_num_ = true;}
    inline void set_channels(int x) { channels_ = x; has_channels_ = true;}
    inline void set_height(int x) { height_ = x; has_height_ = true;}
    inline void set_width(int x) { width_ = x; has_width_ = true;}

    std::vector<double> double_data() const { return double_data_; }
    std::vector<double> double_diff() const { return double_diff_; }
    int double_data_size() const { return double_data_.size(); }
    double double_data(int i) const { return double_data_[i]; }
    int double_diff_size() const { return double_diff_.size(); }
    double double_diff(int i) const { return double_diff_[i]; }

    std::vector<float>  diff() const { return diff_; }
    int diff_size() const { return diff_.size(); }
    float diff(int i) const { return diff_[i]; }
    std::vector<float>  data() const { return data_; }
    int data_size() const { return data_.size();}
    float data(int i) const { return data_[i]; }


    void CopyFrom(const BlobProto& other) {
      num_ = other.num();
      channels_ = other.channels();
      height_ = other.height();
      width_ = other.width();
      shape_.CopyFrom(other.shape_);
      has_num_ = other.has_num();
      has_channels_ = other.has_channels();
      has_height_ = other.has_height();
      has_width_ = other.has_width();
    }
   
  private:
    int num_;
    int channels_;
    int height_;
    int width_;
    BlobShape shape_;

    bool has_num_;
    bool has_channels_;
    bool has_height_;
    bool has_width_;

    std::vector<double> double_data_;
    std::vector<double> double_diff_;
    std::vector<float> diff_;
    std::vector<float> data_;
};
#endif

