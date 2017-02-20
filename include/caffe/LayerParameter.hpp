#ifndef _LAYERPARAMETER_H_
#define _LAYERPARAMETER_H_
#include <string>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>

namespace caffe {

enum Phase {
   TRAIN = 0,
   TEST = 1
};

//template <typename DType>
class LayerParameter {
  public:
    //TODO
    string name() const { return "TODO"; }
    //TODO
    string type() const { return "TODO"; }
    //TODO
    int blobs_size() const { return 0; }
    //TODO
    int loss_weight_size() const {return 0;}
    //TODO
    float loss_weight(int index) const { return (float)0.0; }
    //TODO
    Phase phase() const {return caffe::Phase::TRAIN;}

    void set_name(string name) {
      name_ = name; 
    }
    void set_type(string type) {
      type_ = type;
    }
    int bottom_size() const {
      return bottom_name.size();
    }
    int top_size() const {
      return top_name.size();
    }

    vector<string> bottom_name;
    vector<string> top_name;
  private:
    string name_;
    string type_;

};

}//end caffe
#endif
