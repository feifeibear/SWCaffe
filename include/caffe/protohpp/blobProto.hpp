#ifndef _PROTOBLOB_H_
#define _PROTOBLOB_H_
class BlobProto {
  public:
    BlobProto(int num, int channels, int height, int width):
      num_(num), channels_(channels), height_(height), width_(width)
      {}
    void CopyFrom(const BlobProto& other) {
      num_ = other.num();
      channels_ = other.channels();
      height_ = other.height();
      width_ = other.width();
    }
    bool has_num() const { return has_num_; }
    bool has_channels() const { return has_channels_; }
    bool has_height() const { return has_height_; }
    bool has_width() const { return has_width_; }

    int num() const { return num_; }
    int channels() const { return channels_; }
    int height() const { return height_; }
    int width() const { return width_; }
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

