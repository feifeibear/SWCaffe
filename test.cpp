#include "caffe/caffe.hpp"
using namespace caffe;

int main () {

  std::vector<LayerParameter> layerparams;
  std::vector<std::string> bottom1, top1;
  top1.push_back("data");
  LayerParameter l1("data","Input",bottom1, top1, 0);

  std::vector<std::string> bottom2, top2;
  bottom2.push_back("data"); top2.push_back("conv1");
  LayerParameter l2("conv1","InnerProduct",bottom2, top2, 2);

  std::vector<std::string> bottom3, top3;
  bottom3.push_back("conv1"); top3.push_back("conv3");
  LayerParameter l3("conv2","InnerProduct", bottom3, top3,2);
  //LayerParameter l4("conv2","Convolution",{"pool1"},{"conv2"},2);

  layerparams.push_back(l1);
  layerparams.push_back(l2);
  layerparams.push_back(l3);

  NetParameter net_param("mynet", layerparams);

  Net<float> net(net_param);

  return 0;
}
