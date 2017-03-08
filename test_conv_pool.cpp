#include "caffe/caffe.hpp"
using namespace caffe;

int main () {

  std::vector<LayerParameter> layerparams;
  std::vector<std::string> bottom1, top1;
  top1.push_back("data");
  std::vector<std::vector<int> >input_size;
  input_size.resize(1);
  input_size[0].push_back(2);
  input_size[0].push_back(1);
  input_size[0].push_back(8);
  input_size[0].push_back(8);
  DLOG(INFO) <<  "InputParameter Initialization is OK!";

  InputParameter input_param(input_size);
  LayerParameter l1("data","Input",bottom1, top1, 0);
  l1.setup_input_param(input_param);
  DLOG(INFO) <<  "input layer paramter is OK!";

  //BlobProto w1(10,20,1,1);
  /*InnerProductParameter innerparam1(10);
  std::vector<std::string> bottom2, top2;
  bottom2.push_back("data"); top2.push_back("ip1");
  LayerParameter l2("ip1","InnerProduct",bottom2, top2, 0);
  l2.setup_inner_product_param(innerparam1);
  */
  ConvolutionParameter conv_param(3);
  conv_param.set_pad_2d();
  conv_param.set_kernel_2d(3,3);
  conv_param.set_stride_2d();
  std::vector<std::string> bottom2, top2;
  bottom2.push_back("data"); top2.push_back("conv1");
  LayerParameter l2("conv1", "Convolution", bottom2, top2, 0);
  l2.setup_convolution_param(conv_param);

  PoolingParameter pool_param;
  pool_param.set_pad_2d();
  pool_param.set_kernel_2d(2,2);
  pool_param.set_stride_2d(2,2);
  std::vector<std::string> bottom3, top3;
  bottom3.push_back("conv1"); top3.push_back("pool1");
  LayerParameter l3("pool1", "Pooling", bottom3, top3, 0);
  l3.setup_pooling_param(pool_param);
  /*InnerProductParameter innerparam2(20);
  std::vector<std::string> bottom3, top3;
  bottom3.push_back("ip1"); top3.push_back("ip3");
  LayerParameter l3("ip2","InnerProduct", bottom3, top3, 0);
  l3.setup_inner_product_param(innerparam2);
  //LayerParameter l4("conv2","Convolution",{"pool1"},{"conv2"},2);
  */

  layerparams.push_back(l1);
  layerparams.push_back(l2);
  layerparams.push_back(l3);
  DLOG(INFO) <<  "paramter Initialization is OK!";

  NetParameter net_param("mynet", layerparams);

  Net<float> net(net_param);
  net.visit_for_check();

  //DLOG(INFO) << "begin forward pass of this net";
  //float loss = 0;
  //net.Forward(&loss);

  //DLOG(INFO) << "begin backward pass of this net";
  //net.Backward();

  net.set_debug_info( true  );
  net.fjr_rand_init_input_blobs();
  net.fjr_rand_init_output_blobs();

  DLOG(INFO) << "begin backward pass of this net";
  net.ForwardBackward();

  DLOG(INFO) << "test end";
  return 0;
}
