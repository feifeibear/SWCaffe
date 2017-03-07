#include "caffe/caffe.hpp"
using namespace caffe;

int main () {

  /********
   * define input layer (2, 1, 8, 8) 
   ********/
  std::vector<std::string> bottom1, top1;
  top1.push_back("data");
  std::vector<std::vector<int> >input_size;
  input_size.resize(1);
  input_size[0].push_back(2);
  input_size[0].push_back(1);
  input_size[0].push_back(8);
  input_size[0].push_back(8);
  InputParameter input_param(input_size);
  LayerParameter l1("data","Input",bottom1, top1, 0);
  l1.setup_input_param(input_param);
  DLOG(INFO) <<  "input layer paramter is OK!";


  /********
   * define convolution layer (1,3,3,3) -> (3, 1, 6, 6)
   *******/
  ConvolutionParameter conv_param(3);
  conv_param.set_pad_2d();
  conv_param.set_kernel_2d(3,3);
  conv_param.set_stride_2d();
  std::vector<std::string> bottom2, top2;
  bottom2.push_back("data"); top2.push_back("conv1");
  LayerParameter l2("conv1", "Convolution", bottom2, top2, 0);
  l2.setup_convolution_param(conv_param);


  /*****
   * define pool layer -> (3, 1, 3, 3)
   * ****/
  PoolingParameter pool_param;
  pool_param.set_pad_2d();
  pool_param.set_kernel_2d(2,2);
  pool_param.set_stride_2d();
  std::vector<std::string> bottom3, top3;
  bottom3.push_back("conv1"); top3.push_back("pool1");
  LayerParameter l3("pool1", "Pooling", bottom3, top3, 0);
  l3.setup_pooling_param(pool_param);

  /******
   * define InnerProduct layer (27) -> (3) 
   * ****/
  InnerProductParameter innerparam1(3);
  std::vector<std::string> bottom4, top4;
  bottom4.push_back("pool1"); top4.push_back("ip1");
  LayerParameter l4("ip1","InnerProduct",bottom4, top4, 0);
  l4.setup_inner_product_param(innerparam1);


  /******
   * define InnerProduct layer (3) -> (4)
   * ****/
  InnerProductParameter innerparam2(4);
  std::vector<std::string> bottom5, top5;
  bottom5.push_back("ip1"); top5.push_back("ip2");
  LayerParameter l5("ip2","InnerProduct", bottom5, top5, 0);
  l5.setup_inner_product_param(innerparam2);
  //LayerParameter l4("conv2","Convolution",{"pool1"},{"conv2"},2);

  std::vector<LayerParameter> layerparams;
  layerparams.push_back(l1);
  layerparams.push_back(l2);
  layerparams.push_back(l3);
  layerparams.push_back(l4);
  layerparams.push_back(l5);
  DLOG(INFO) <<  "paramter Initialization is OK!";

  NetParameter net_param("mynet", layerparams);

  Net<float> net(net_param);
  net.visit_for_check();

  DLOG(INFO) << "begin forward pass of this net";
  float loss = 0;
  //net.Forward(&loss);

  DLOG(INFO) << "begin backward pass of this net";
  //net.Backward();

  //set backward

  DLOG(INFO) << "begin backward pass of this net";
  net.fjr_rand_init_output_blobs();
  net.fjr_rand_init_input_blobs();

  net.ForwardBackward();

  DLOG(INFO) << "test end";
  return 0;
}
