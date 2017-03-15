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
  input_size[0].push_back(1);
  input_size[0].push_back(1);
  input_size[0].push_back(28);
  input_size[0].push_back(28);
  InputParameter input_param(input_size);
  LayerParameter l1("data","Input",bottom1, top1, 0);
  l1.setup_input_param(input_param);
  DLOG(INFO) <<  "input layer paramter is OK!";
*/

  
  //mnist input 10, 1, 28, 28
  std::vector<std::string> bottom1, top1;
  top1.push_back("data");
  top1.push_back("label");
  DataParameter data_param;
  data_param.set_batch_size(10);
  LayerParameter l1("data", "Data", bottom1, top1, 0);
  l1.setup_data_param(data_param);
  DLOG(INFO) <<  "Data layer paramter is OK!";

  /********
   * define convolution layer (10,1,28,28) -> (10, 20, 24, 24)
   *******/
  ConvolutionParameter conv_param1(20);
  conv_param1.mutable_weight_filler().set_type("xavier");
  conv_param1.mutable_bias_filler().set_type("constant");
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
  conv_param2.mutable_weight_filler().set_type("xavier");
  conv_param2.mutable_bias_filler().set_type("constant");
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
  innerparam1.mutable_weight_filler().set_type("xavier");
  innerparam1.mutable_bias_filler().set_type("constant");
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
  innerparam2.mutable_weight_filler().set_type("xavier");
  innerparam2.mutable_bias_filler().set_type("constant");
  std::vector<std::string> bottom8, top8;
  bottom8.push_back("relu1"); top8.push_back("ip2");
  LayerParameter l8("ip2", "InnerProduct", bottom8, top8, 0);
  l8.setup_inner_product_param(innerparam2);


  LossParameter loss_param;
  std::vector<std::string> bottom9, top9;
  bottom9.push_back("ip2"); bottom9.push_back("label"); top9.push_back("loss");
  LayerParameter l9("loss", "SoftmaxWithLoss", bottom9, top9, 0);
  l9.setup_loss_param(loss_param);
  // softmax_with_loss layer use scalar labels y, since loss=-log(softmax(y)[x]). (Euclidean use vec)

  AccuracyParameter accuracy_param;
  std::vector<std::string> bottom10, top10;
  bottom10.push_back("ip2"); bottom10.push_back("label"); top10.push_back("accuracy");
  LayerParameter l10("accuracy", "Accuracy", bottom10, top10, 0);
  l10.setup_accuracy_param(accuracy_param);
  l10.add_include(TEST);


  std::vector<LayerParameter> layerparams;
  layerparams.push_back(l1);
  layerparams.push_back(l2);
  layerparams.push_back(l3);
  layerparams.push_back(l4);
  layerparams.push_back(l5);
  layerparams.push_back(l6);
  layerparams.push_back(l7);
  layerparams.push_back(l8);
  layerparams.push_back(l9);
  layerparams.push_back(l10);
  DLOG(INFO) <<  "paramter Initialization is OK!";

  NetParameter net_param("mynet", layerparams);
  DLOG(INFO) <<  "net paramter Initialization is OK!";


  Net<float> net(net_param);
  net.set_debug_info(true);
  //net.fjr_rand_init_input_blobs();

  DLOG(INFO) << "Init solver_param...";
  SolverParameter solver_param(10, 100, 0.01, 0, 10000, "fixed", 0.9, 0.0005);
  DLOG(INFO) << "Set net_param...";
  solver_param.set_net(net_param);
  DLOG(INFO) << "Init solver...";
  shared_ptr<Solver<float> >
      solver(SolverRegistry<float>::CreateSolver(solver_param));
  DLOG(INFO) << "Begin solve...";
  solver->Solve(NULL);

  /*DLOG(INFO) << "begin forward pass of this net";
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
*/

  DLOG(INFO) << "test end";
  return 0;
}



/*#include "caffe/caffe.hpp"
using namespace caffe;

int main () {

  std::vector<LayerParameter> layerparams;
  std::vector<std::string> bottom1, top1;
  top1.push_back("data");
  std::vector<std::vector<int> >input_size;
  input_size.resize(1);
  input_size[0].push_back(10);
  input_size[0].push_back(1);
  input_size[0].push_back(8);
  input_size[0].push_back(8);
  DLOG(INFO) <<  "InputParameter Initialization is OK!";

  InputParameter input_param(input_size);
  LayerParameter l1("data","Input",bottom1, top1, 0);
  l1.setup_input_param(input_param);
  DLOG(INFO) <<  "input layer paramter is OK!";

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

  layerparams.push_back(l1);
  layerparams.push_back(l2);
  layerparams.push_back(l3);
  DLOG(INFO) <<  "paramter Initialization is OK!";

  NetParameter net_param("mynet", layerparams);

  Net<float> net(net_param);
  net.visit_for_check();

  
  /**explicit SolverParameter(int test_iter, int test_interval, float base_lr, int display,
      int max_iter, std::string lr_policy, float momentum, float weight_decay,
      std::string type="SGD", int num_test=1)
  
  DLOG(INFO) << "Init solver_param...";
  SolverParameter solver_param(10, 10, 0.1, 0, 1000, "fixed", 0.9, 0.005);
  DLOG(INFO) << "Set net_param...";
  solver_param.set_net(net_param);
  DLOG(INFO) << "Init solver...";
  shared_ptr<Solver<float> >
      solver(SolverRegistry<float>::CreateSolver(solver_param));
  DLOG(INFO) << "Begin solve...";
  solver->Solve(NULL);
  /*DLOG(INFO) << "begin forward pass of this net";
  float loss = 0;
  net.Forward(&loss);

  DLOG(INFO) << "begin backward pass of this net";
  net.Backward();

  DLOG(INFO) << "begin backward pass of this net";
  net.ForwardBackward();

  DLOG(INFO) << "test end";
  return 0;
}
*/
