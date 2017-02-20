#ifndef SWNET_NET_HPP_
#define SWNET_NET_HPP_
#include <vector>
#include <base_layer.h>

class NetParameter {
public:
  vector<int> B_array;
  vector<int> N_array;
  vector<int> C_array;
  vector<int> R_array;
  std::string type_;
};
class Net {
  void Init(const vector<NetParameter>& params){
    std::string type = params.type_;
    switch(type){
      case "conv" :
        break;
      case "relu" : 
        break;
      case "data" :
        break;
      case "mlp" :
        break;
      default:
        break;
    }
  }

protected:
  string name_;
  vector<shared_ptr<layer> > layers_;
  vector<shared_ptr<tensor> > tensors_;
  vector< vector<tensor*> > bottom_vecs_;
  vector< vector<tensor*> > top_vecs_;
};
#endif  // _NET_HPP_
