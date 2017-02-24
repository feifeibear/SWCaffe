#ifndef _INPUTPARAMETER_H_
#define _INPUTPARAMETER_H_

namespace caffe {

class InputParameter{
  public:
    InputParameter() {}
    explicit InputParameter(std::vector<std::vector<int> > shapes){
      int len1 = shapes.size();
      shapes_.resize(len1);
      for( int i = 0; i < len1; ++i ){
        int len2 = shapes[i].size();
        shapes_[i].resize(len2);
        for( int j = 0; j < len2; ++j )
          shapes_[i][j] = shapes[i][j];
      }
    }
    inline void CopyFrom(const InputParameter& other){
      int len1 = other.shape_size();
      shapes_.resize(len1);
      for( int i = 0; i < len1; ++i ){
        int len2 = other.get_shape()[i].size();
        shapes_[i].resize(len2);
        for( int j = 0; j < len2; ++j )
          shapes_[i][j] = other.get_shape()[i][j];
      }
    }
    inline int shape_size() const { return shapes_.size(); }
    inline const std::vector<std::vector<int> > get_shape() const { return shapes_; }
    inline const std::vector<int> shape(int id) const { return shapes_[id]; }
  public:
    std::vector<std::vector<int> > shapes_;
};

}
#endif
