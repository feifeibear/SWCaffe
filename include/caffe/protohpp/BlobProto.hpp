#ifndef _PROTOBLOB_H_
#define _PROTOBLOB_H_

class BlobShape {
public:
  BlobShape() {}
  void set_dim(std::vector<int> dim) { dim_ = dim; }
  int dim(int i) const { return dim_[i]; }
  std::vector<int> dim_vec() const { return dim_; }
  int dim_size() const { return dim_.size(); }
  void CopyFrom(const BlobShape& other) {
    dim_ = other.dim_vec();
  }
private:
  std::vector<int> dim_;
};

class BlobProto {
  public:
    BlobProto(int num, int channels, int height, int width):
      num_(num), channels_(channels), height_(height), width_(width)
      {
        std::vector<int> d;
        d.resize(4);
        d[0] = num_; d[1] = channels_;
        d[2] = height_; d[3] = width_;
        shape_.set_dim(d);
      }
    void CopyFrom(const BlobProto& other) {
      num_ = other.num();
      channels_ = other.channels();
      height_ = other.height();
      width_ = other.width();
      shape_.CopyFrom(other.shape_);
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

