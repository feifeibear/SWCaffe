#ifndef _INPUTPARAMETER_H_
#define _INPUTPARAMETER_H_

namespace caffe {

class InputParameter{
  public:
    InputParameter() {}

    InputParameter(const InputParameter& other){
      this->CopyFrom(other);
    }

    inline InputParameter& operator=(const InputParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    inline void CopyFrom(const InputParameter& other){
      int shape_size = other.shape_size();
      shapes_.resize(shape_size);
      for (int i=0; i<shape_size; i++)
        shapes_[i].CopyFrom(other.shape(i));
    }

    inline int shape_size() const { return shapes_.size(); }
    inline const BlobShape& shape(int id) const { return shapes_[id]; }
    inline void add_shape(const BlobShape& x) { 
      BlobShape b(x);
      shapes_.push_back(b); 
    }

  public:
    std::vector<BlobShape> shapes_;
};

}
#endif
