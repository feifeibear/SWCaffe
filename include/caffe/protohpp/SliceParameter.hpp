#ifndef _SLICEPARAMETER_
#define _SLICEPARAMETER_

#include <vector>

namespace caffe {

class SliceParameter {
  public:
    SliceParameter() {
      axis_ = 1;
      slice_dim_ = 1;
      has_slice_dim_ = false;
      has_axis_ = false;
    }
    
    SliceParameter(const SliceParameter& other){
      this->CopyFrom(other);
    }

    inline SliceParameter& operator=(const SliceParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const SliceParameter& other ) {
      axis_ = other.axis();
      slice_point_ = other.slice_point();
      slice_dim_ = other.slice_dim();
      has_slice_dim_ = other.has_slice_dim();
      has_axis_ = other.has_axis();
    }

    inline int axis() const { return axis_; }
    inline std::vector<int> slice_point() const { return slice_point_; }
    inline int slice_point(int i) const { return slice_point_[i]; }
    inline int slice_point_size() const { return slice_point_.size(); }
    inline int slice_dim() const { return slice_dim_; }
    inline bool has_slice_dim() const { return has_slice_dim_; }
    inline bool has_axis() const { return has_axis_; }

    inline void set_axis(int x) { has_axis_ = true; axis_ = x; }
    inline void add_slice_point(int x) {slice_point_.push_back(x);}
    inline void set_slice_dim(int x) { has_slice_dim_ = true; slice_dim_=x;}

    static SliceParameter default_instance_;
    
  private:
    int axis_;
    std::vector<int> slice_point_;
    int slice_dim_;

    bool has_slice_dim_, has_axis_;
};

}

#endif
