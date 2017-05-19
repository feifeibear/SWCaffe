#include "caffe/caffe.hpp"
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
using namespace caffe;

class Serial_Blob
{
public:
  std::vector<float> data;
  std::vector<float> diff;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & data;
    ar & diff;
  }
};

class Serial_Layer
{
public:
  std::vector<Serial_Blob> blobs;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & blobs;
  }
};

class Serial_Net
{
public:
  std::vector<Serial_Layer> layers;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & layers;
  }
};

int main()
{
  NetParameter net_param;
  ReadNetParamsFromBinaryFileOrDie("../data/VGG_ILSVRC_16_layers.caffemodel", &net_param);

  Serial_Net net;
  int num_layer = net_param.layer_size();
  net.layers.resize(num_layer);
  for (int i=0; i<num_layer; i++){
    int num_blob = net_param.layer(i).blobs_size();
    net.layers[i].blobs.resize(num_blob);
    for (int j=0; j<num_blob; j++){
      int num_data = net_param.layer(i).blobs(j).data_size();
      net.layers[i].blobs[j].data.resize(num_data);
      for (int k=0; k<num_data; k++)
        net.layers[i].blobs[j].data[k] = net_param.layer(i).blobs(j).data(k);
      int num_diff = net_param.layer(i).blobs(j).diff_size();
      net.layers[i].blobs[j].diff.resize(num_diff);
      for (int k=0; k<num_diff; k++)
        net.layers[i].blobs[j].diff[k] = net_param.layer(i).blobs(j).diff(k);
    }
  }

  std::ofstream ofs("serialized_caffemodel");
  {
    boost::archive::text_oarchive oa(ofs);
    oa << net;
  }

	return 0;
}