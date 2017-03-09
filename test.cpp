#include "caffe/caffe.hpp"
using namespace caffe;

int main () {

  /********
   * define input layer (2, 1, 8, 8) 
   ********/

  /*
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
  */

  //mnist input 10, 1, 28, 28
  std::vector<std::string> bottom1, top1;
  top1.push_back("data");
  DataParameter data_param(10);
  LayerParameter l1("data", "Data", bottom1, top1, 0);
  l1.setup_data_param(data_param);
  DLOG(INFO) <<  "Data layer paramter is OK!";

  /********
   * define convolution layer (10,1,28,28) -> (10, 20, 24, 24)
   *******/
  ConvolutionParameter conv_param1(20);
  conv_param1.set_pad_2d();
  conv_param1.set_kernel_2d(5,5);
  conv_param1.set_stride_2d(1,1);
  std::vector<std::string> bottom2, top2;
  bottom2.push_back("data"); top2.push_back("conv1");
  LayerParameter l2("conv1", "Convolution", bottom2, top2, 0);
  l2.setup_convolution_param(conv_param1);


  /*****
   * define pool layer -> (10, 20, 12, 12)
   * ****/
  PoolingParameter pool_param1;
  pool_param1.set_pad_2d();
  pool_param1.set_kernel_2d(2,2);
  pool_param1.set_stride_2d(2,2);
  std::vector<std::string> bottom3, top3;
  bottom3.push_back("conv1"); top3.push_back("pool1");
  LayerParameter l3("pool1", "Pooling", bottom3, top3, 0);
  l3.setup_pooling_param(pool_param1);

  /********
   * define convolution layer (10,20,12,12) -> (10, 50, 8, 8)
   *******/
  ConvolutionParameter conv_param2(50);
  conv_param2.set_pad_2d();
  conv_param2.set_kernel_2d(5,5);
  conv_param2.set_stride_2d(1,1);
  std::vector<std::string> bottom4, top4;
  bottom4.push_back("pool1"); top4.push_back("conv2");
  LayerParameter l4("conv2", "Convolution", bottom4, top4, 0);
  l4.setup_convolution_param(conv_param2);


  /*****
   * define pool layer -> (10, 50, 4, 4)
   * ****/
  PoolingParameter pool_param2;
  pool_param2.set_pad_2d();
  pool_param2.set_kernel_2d(2,2);
  pool_param2.set_stride_2d(2,2);
  std::vector<std::string> bottom5, top5;
  bottom5.push_back("conv2"); top5.push_back("pool2");
  LayerParameter l5("pool2", "Pooling", bottom5, top5, 0);
  l5.setup_pooling_param(pool_param2);

  /******
   * define InnerProduct layer (50*4*4) -> (500) 
   * ****/
  InnerProductParameter innerparam1(500);
  std::vector<std::string> bottom6, top6;
  bottom6.push_back("pool2"); top6.push_back("ip1");
  LayerParameter l6("ip1","InnerProduct",bottom6, top6, 0);
  l6.setup_inner_product_param(innerparam1);


  ReLUParameter reluparam1;
  std::vector<std::string> bottom7, top7;
  bottom7.push_back("ip1"); top7.push_back("relu1");
  LayerParameter l7("relu1", "ReLU", bottom7, top7, 0);
  l7.setup_relu_param(reluparam1);

  /******
   * define InnerProduct layer (500) -> (10)
   * ****/
  InnerProductParameter innerparam2(10);
  std::vector<std::string> bottom8, top8;
  bottom8.push_back("relu1"); top8.push_back("ip2");
  LayerParameter l8("ip2", "InnerProduct", bottom8, top8, 0);
  l8.setup_inner_product_param(innerparam2);
  //LayerParameter l4("conv2","Convolution",{"pool1"},{"conv2"},2);

  std::vector<LayerParameter> layerparams;
  layerparams.push_back(l1);
  layerparams.push_back(l2);
  layerparams.push_back(l3);
  layerparams.push_back(l4);
  layerparams.push_back(l5);
  layerparams.push_back(l6);
  layerparams.push_back(l7);
  DLOG(INFO) <<  "paramter Initialization is OK!";

  NetParameter net_param("mynet", layerparams);
  DLOG(INFO) <<  "net paramter Initialization is OK!";


  Net<float> net(net_param);
  net.set_debug_info(true);
  net.visit_for_check();

  DLOG(INFO) << "begin forward pass of this net";
  float loss = 0;
  //net.Forward(&loss);

  DLOG(INFO) << "begin backward pass of this net";
  //net.Backward();

  //set backward

  DLOG(INFO) << "begin backward pass of this net";
  net.fjr_rand_init_output_blobs();
  //net.fjr_rand_init_input_blobs();

  net.ForwardBackward();
  net.ForwardBackward();

  DLOG(INFO) << "test end";
  return 0;
}
