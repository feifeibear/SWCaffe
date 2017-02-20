#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <set>
#include "shared_ptr/shared_ptr.hpp"
#include <glog/logging.h>

#include "blob.hpp"

template<typename T>
class Layer{
public:
  std::string name_;
  std::string top_;
  std::string bottom_;
  //type
  std::string type_;
};

class LayerParam{
  public:
    void set_type(std::string type){
      type_ = type;
    }
    void set_name(std::string name) {
      name_ = name;
    } 
    std::string type() const {
      return type_;
    }
    std::string name() const {
      return name_;
    }

    std::vector<std::string> bottom_name;
    std::vector<std::string> top_name;

    int top_size() const {return top_name.size();}
    int bottom_size() const {return bottom_name.size();}
  private:
    std::string type_;
    std::string name_;
};

class NetParameters {
public:
  std::string name() const {return name_;}
  void set_name(std::string name) {name_ = name;}
  int layer_size() const {return layer.size();}
  std::vector<LayerParam> layer;
private:
  std::string name_;
};

//TODO define layers_
template<typename T>
class Net {
public:
  void Init(const NetParameters& params) {
    name_ = params.name();
    std::map<std::string, int> blob_name_to_idx;
    std::set<std::string> available_blobs;

    //for each layer, set up its input and output
    int layer_size = params.layer_size();
    top_vecs_.resize(layer_size);
    top_id_vecs_.resize(layer_size);
    bottom_vecs_.resize(layer_size);
    bottom_id_vecs_.resize(layer_size);

    for (int layer_id = 0; layer_id < layer_size; ++layer_id) {
      const LayerParam& layer_param = params.layer[layer_id];

      //add new layer to network
      //TODO use CreateLayer
      shared_ptr<Layer<T> > new_layer(new Layer<T>());
      layers_.push_back(new_layer);
      layer_names_.push_back(layer_param.name());


      //add bottom blobs to layers
      for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); bottom_id++){
        const std::string& blob_name = layer_param.bottom_name[bottom_id];
        const int blob_id = blob_name_to_idx[blob_name];
        bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
        bottom_id_vecs_[layer_id].push_back(blob_id);
        std::cout <<"erase "<< blob_name << std::endl;
        available_blobs.erase(blob_name);
      }

      //add top layers
      for (int top_id = 0; top_id < layer_param.top_size(); ++top_id) {
        //get top name
        const std::string& blob_name = layer_param.top_name[top_id];
        //inline computing
        if(layer_param.bottom_name.size() > top_id &&
            layer_param.bottom_name[top_id] == blob_name) {
          top_vecs_[layer_id].push_back(blobs_[blob_name_to_idx[blob_name]].get());
          top_id_vecs_[layer_id].push_back(blob_name_to_idx[blob_name]);
        } else{
          //new a Blob, assign an id, id - name
          shared_ptr<Blob<T> > blob_pointer(new Blob<T>(blob_name));
          const int blob_id = blobs_.size();
          blobs_.push_back(blob_pointer);
          blob_name_to_idx[blob_name] = blob_id;

          //connect its top and bottom
          top_id_vecs_[layer_id].push_back(blob_id);
          top_vecs_[layer_id].push_back(blob_pointer.get());

          //collect input layer tops as Net inputs
          if(layer_param.type() == "Input") {
            const int blob_id = blobs_.size()-1;
            net_input_blob_indices.push_back(blob_id);
            net_input_blobs_.push_back(blobs_[blob_id].get());
          }
        }
        std::cout <<"insert "<< blob_name << std::endl; 
        available_blobs.insert(blob_name);
      }
      //TODO set up layer

      std::cout << "layer id " << layer_id <<" allocate blob is OK!" << std::endl;
    }
//In the end, all remaining avalibale blobs are considered output blobs
    for(std::set<std::string>::iterator it = available_blobs.begin();
        it != available_blobs.end(); ++it) {
      net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
      net_output_blob_indices.push_back(blob_name_to_idx[*it]);
    }
  }

  void visit_for_check(){
    std::cout << "* print blobs_" <<std::endl;
    for(int i = 0; i < blobs_.size(); ++i){
      std::cout << blobs_[i]->name() << " ";
    }
    std::cout << std::endl;

    std::cout << "* bottom_id_vecs_" <<std::endl;
    for(int i = 0; i < bottom_vecs_.size(); ++i){
      std::cout << i << " : ";
      for(int j = 0; j < bottom_vecs_[i].size(); ++j)
        std::cout << bottom_vecs_[i][j]->name() << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "* top_id_vecs_" <<std::endl;
    for(int i = 0; i < top_vecs_.size(); ++i){
      std::cout << i << " : ";
      for(int j = 0; j < top_vecs_[i].size(); ++j)
        std::cout << top_vecs_[i][j]->name() << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "* input blobs" <<std::endl;
    for(int i = 0; i < net_input_blobs_.size(); ++i){
      std::cout << net_input_blobs_[i]->name() << " ";
    }
    std::cout << std::endl;

    std::cout << "* output blobs" <<std::endl;
    for(int i = 0; i < net_output_blobs_.size(); ++i){
      std::cout << net_output_blobs_[i]->name() << " ";
    }
    std::cout << std::endl;

  }
  std::string name() const { return name_;  }

private:
  std::string name_;
  //intermediate data between layers
  std::vector<shared_ptr<Blob<T> > > blobs_;
  std::vector<std::vector<int> > blobs_id_;

  std::vector<shared_ptr<Layer<T> > > layers_;
  std::vector<std::string> layer_names_;

  //input for each layers
  std::vector<std::vector<Blob<T>*> > bottom_vecs_;
  std::vector<std::vector<int> > bottom_id_vecs_;
  //output for each layer
  std::vector<std::vector<Blob<T>*> > top_vecs_;
  std::vector<std::vector<int> > top_id_vecs_;

  //input for Net
  std::vector<int> net_input_blob_indices;
  std::vector<int> net_output_blob_indices;
  std::vector<Blob<T>*> net_input_blobs_;
  std::vector<Blob<T>*> net_output_blobs_;
  //params

};

int main() {
  NetParameters netparam;
  netparam.set_name("my_net");

  //set param of each layer
  LayerParam l1;
  l1.set_type("Input");
  l1.set_name("l1");
  l1.top_name.push_back("data");

  LayerParam l2;
  l2.set_type("Convolution");
  l2.set_name("conv1");
  l2.bottom_name.push_back("data");
  l2.top_name.push_back("conv1");

  LayerParam l3;
  l3.set_type("Pooling");
  l3.set_name("pool1");
  l3.bottom_name.push_back("conv1");
  l3.top_name.push_back("pool1");

  LayerParam l4;
  l4.set_type("InnerProduct");
  l4.set_name("ip1");
  l4.bottom_name.push_back("pool1");
  l4.top_name.push_back("ip1");

  LayerParam l5;
  l5.set_type("ReLU");
  l5.set_name("relu1");
  l5.bottom_name.push_back("ip1");
  l5.top_name.push_back("ip1");

  LayerParam l6;
  l6.set_type("InnerProduct");
  l6.set_name("ip2");
  l6.bottom_name.push_back("ip1");
  l6.top_name.push_back("ip2");

  LayerParam l7;
  l7.set_type("Softmax");
  l7.set_name("prob");
  l7.bottom_name.push_back("ip2");
  l7.top_name.push_back("prob");


  netparam.layer.push_back(l1);
  netparam.layer.push_back(l2);
  netparam.layer.push_back(l3);
  netparam.layer.push_back(l4);
  netparam.layer.push_back(l5);
  netparam.layer.push_back(l6);
  netparam.layer.push_back(l7);

  //init Net
  Net<float> my_net;
  my_net.Init(netparam);
  my_net.visit_for_check();

  LOG(INFO) << "OK";

  return 0;
}

