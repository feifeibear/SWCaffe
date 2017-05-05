#include "caffe/caffe.hpp"
#include "caffe/util/serialize.hpp"
#include <iostream>
#include <fstream>
using namespace caffe;

int main()
{
  NetParameter net_param;
  ReadNetParamsFromBinaryFileOrDie("../data/VGG_ILSVRC_16_layers.caffemodel", &net_param);

  Serial_Net net;
  int num_layer = net_param.layer_size();
  net.layers.resize(num_layer);
  for (int i=0; i<num_layer; i++){
    net.layers[i].name = net_param.layer(i).name();
    int num_blob = net_param.layer(i).blobs_size();
    net.layers[i].blobs.resize(num_blob);
    for (int j=0; j<num_blob; j++){
      int num_data = net_param.layer(i).blobs(j).data_size();
      net.layers[i].blobs[j].data.resize(num_data);
      for (int k=0; k<num_data; k++)
        net.layers[i].blobs[j].data[k] = net_param.layer(i).blobs(j).data(k);
    }
  }

  std::ofstream ofs("../data/serialized_caffemodel", std::ios::out | std::ios::binary);
  net.serialize_out(ofs);
  ofs.close();

  return 0;
}