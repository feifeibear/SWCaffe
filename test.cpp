#include "caffe/caffe.hpp"
using namespace caffe;

int main () {

  std::vector<LayerParameter> layerparams;
  std::vector<std::string> bottom1, top1;
  top1.push_back("data");
  std::vector<std::vector<int> >input_size;
  input_size.resize(1);
  input_size[0].push_back(2);
  input_size[0].push_back(3);
  input_size[0].push_back(4);
  input_size[0].push_back(5);
  DLOG(INFO) <<  "InputParameter Initialization is OK!";

  InputParameter input_param(input_size);
  LayerParameter l1("data","Input",bottom1, top1, 0);
  l1.setup_input_param(input_param);
  DLOG(INFO) <<  "input layer paramter is OK!";

  InnerProductParameter innerparam1(10);
  std::vector<std::string> bottom2, top2;
  bottom2.push_back("data"); top2.push_back("conv1");
  LayerParameter l2("conv1","InnerProduct",bottom2, top2, 2);
  l2.setup_inner_product_param(innerparam1);


  InnerProductParameter innerparam2(20);
  std::vector<std::string> bottom3, top3;
  bottom3.push_back("conv1"); top3.push_back("conv3");
  LayerParameter l3("conv2","InnerProduct", bottom3, top3,2);
  l3.setup_inner_product_param(innerparam2);
  //LayerParameter l4("conv2","Convolution",{"pool1"},{"conv2"},2);

  layerparams.push_back(l1);
  layerparams.push_back(l2);
  layerparams.push_back(l3);

  NetParameter net_param("mynet", layerparams);

  Net<float> net(net_param);
  net.visit_for_check();

  return 0;
}
