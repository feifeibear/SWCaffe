#ifndef _RESHAPEPARAMETER_
#define _RESHAPEPARAMETER_

namespace caffe {

class ReshapeParameter {
  public:
    ReshapeParameter() {
      axis_ = 0;
      num_axes_ = -1;
    }
    
    ReshapeParameter(const ReshapeParameter& other){
      this->CopyFrom(other);
    }

    inline ReshapeParameter& operator=(const ReshapeParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom( const ReshapeParameter& other ) {
      shape_.CopyFrom(other.shape());
      axis_ = other.axis();
      num_axes_ = other.num_axes();
    }

    inline const BlobShape& shape() const { return shape_; }
    inline int axis() const { return axis_; }
    inline int num_axes() const { return num_axes_; }

    inline BlobShape* mutable_shape() { return &shape_; }
    inline void set_axis(int x) { axis_ = x; }
    inline void set_num_axes(int x) {num_axes_=x;}
    
  private:
    BlobShape shape_;
    int axis_;
    int num_axes_;
};

}

#endif
