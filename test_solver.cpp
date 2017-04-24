#include "caffe/caffe.hpp"
#ifdef MYMPI
#include <mpi.h>
#endif
using namespace caffe;

int main (int argc, char ** argv) {
#ifdef MYMPI
  MPI_Init(&argc, &argv);
#endif

  //mnist input 10, 1, 28, 28
  DataParameter data_param_data;
  data_param_data.set_source("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
  data_param_data.set_batch_size(64/Caffe::solver_count());
  //data_param_data.set_batch_size(8);
  LayerParameter data_train;
  data_train.set_name("data_train");
  data_train.set_type("Data");
  data_train.add_top("data");
  data_train.add_top("label");
  data_train.setup_data_param(data_param_data);
  NetStateRule train_include;
  train_include.set_phase(TRAIN);
  data_train.add_include(train_include);

  DataParameter data_param_label;
  data_param_label.set_source("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");
  data_param_label.set_batch_size(128);
  LayerParameter data_test;
  data_test.set_name("data_test");
  data_test.set_type("Data");
  data_test.add_top("data");
  data_test.add_top("label");
  data_test.setup_data_param(data_param_label);
  NetStateRule test_include;
  test_include.set_phase(TEST);
  data_test.add_include(test_include);

  /********
   * define convolution layer (10,1,28,28) -> (10, 20, 24, 24)
   *******/
  ConvolutionParameter conv_param1;
  conv_param1.set_num_output(128);
  conv_param1.set_pad_h(0);
  conv_param1.set_pad_w(0);
  conv_param1.set_kernel_h(5);
  conv_param1.set_kernel_w(5);
  conv_param1.set_stride_h(1);
  conv_param1.set_stride_w(1);
  conv_param1.mutable_weight_filler()->set_type("xavier");
  conv_param1.mutable_bias_filler()->set_type("constant");
  LayerParameter conv1;
  conv1.set_name("conv1");
  conv1.set_type("Convolution");
  conv1.add_bottom("data");
  conv1.add_top("conv1");
  conv1.setup_convolution_param(conv_param1);


  /*****
   * define pool layer -> (10, 20, 12, 12)
   * ****/
  PoolingParameter pool_param1;
  pool_param1.set_pad(0);
  pool_param1.set_kernel_size(2);
  pool_param1.set_stride(2);
  LayerParameter pool1;
  pool1.set_name("pool1");
  pool1.set_type("Pooling");
  pool1.add_bottom("conv1");
  pool1.add_top("pool1");
  pool1.setup_pooling_param(pool_param1);

  /********
   * define convolution layer (10,20,12,12) -> (10, 50, 8, 8)
   *******/
  ConvolutionParameter conv_param2;
  conv_param2.set_num_output(128);
  conv_param2.set_pad_h(0);
  conv_param2.set_pad_w(0);
  conv_param2.set_kernel_h(5);
  conv_param2.set_kernel_w(5);
  conv_param2.set_stride_h(1);
  conv_param2.set_stride_w(1);
  conv_param2.mutable_weight_filler()->set_type("xavier");
  conv_param2.mutable_bias_filler()->set_type("constant");
  LayerParameter conv2;
  conv2.set_name("conv2");
  conv2.set_type("Convolution");
  conv2.add_bottom("pool1");
  conv2.add_top("conv2");
  conv2.setup_convolution_param(conv_param2);

  /*****
   * define pool layer -> (10, 50, 4, 4)
   * ****/
  PoolingParameter pool_param2;
  pool_param2.set_pad(0);
  pool_param2.set_kernel_size(2);
  pool_param2.set_stride(2);
  LayerParameter pool2;
  pool2.set_name("pool2");
  pool2.set_type("Pooling");
  pool2.add_bottom("conv2");
  pool2.add_top("pool2");
  pool2.setup_pooling_param(pool_param2);

  /******
   * define InnerProduct layer (50*4*4) -> (500) 
   * ****/
  InnerProductParameter ip_param1;
  ip_param1.set_num_output(500);
  ip_param1.mutable_weight_filler()->set_type("xavier");
  ip_param1.mutable_bias_filler()->set_type("constant");
  LayerParameter ip1;
  ip1.set_name("ip1");
  ip1.set_type("InnerProduct");
  ip1.add_bottom("pool2");
  ip1.add_top("ip1");
  ip1.setup_inner_product_param(ip_param1);

  ReLUParameter relu_param1;
  LayerParameter relu1;
  relu1.set_name("relu1");
  relu1.set_type("ReLU");
  relu1.add_bottom("ip1");
  relu1.add_top("relu1");
  relu1.setup_relu_param(relu_param1);

  /******
   * define InnerProduct layer (500) -> (10)
   * ****/
  InnerProductParameter ip_param2;
  ip_param2.set_num_output(10);
  ip_param2.mutable_weight_filler()->set_type("xavier");
  ip_param2.mutable_bias_filler()->set_type("constant");
  LayerParameter ip2;
  ip2.set_name("ip2");
  ip2.set_type("InnerProduct");
  ip2.add_bottom("relu1");
  ip2.add_top("ip2");
  ip2.setup_inner_product_param(ip_param2);


  LossParameter loss_param;
  LayerParameter loss;
  loss.set_name("loss");
  loss.set_type("SoftmaxWithLoss");
  loss.add_bottom("ip2");
  loss.add_bottom("label");
  loss.add_top("loss");
  loss.setup_loss_param(loss_param);

  AccuracyParameter accuracy_param;
  LayerParameter accuracy;
  accuracy.set_name("accuracy");
  accuracy.set_type("Accuracy");
  accuracy.add_bottom("ip2");
  accuracy.add_bottom("label");
  accuracy.add_top("accuracy");
  accuracy.setup_accuracy_param(accuracy_param);
  accuracy.add_include(test_include);

  LOG_IF(INFO, Caffe::root_solver())
    <<  "layer paramter Initialization is OK!";

  NetParameter net_param;
  net_param.set_name("lenet");
  net_param.add_layer(data_train);
  net_param.add_layer(data_test);
  net_param.add_layer(conv1);
  net_param.add_layer(pool1);
  net_param.add_layer(conv2);
  net_param.add_layer(pool2);
  net_param.add_layer(ip1);
  net_param.add_layer(relu1);
  net_param.add_layer(ip2);
  net_param.add_layer(loss);
  net_param.add_layer(accuracy);

  LOG_IF(INFO, Caffe::root_solver())
   <<  "net paramter Initialization is OK!";

  //Net<float> net(net_param);
  //net.set_debug_info(true);

  SolverParameter solver_param;
  solver_param.set_net_param(net_param);
  solver_param.add_test_iter(100);
  //solver_param.set_test_interval(500);
  solver_param.set_test_interval(1);
  solver_param.set_base_lr(0.01);
  solver_param.set_display(100);
  //solver_param.set_display(1);
  solver_param.set_max_iter(10000);
  //solver_param.set_max_iter(5);
  solver_param.set_lr_policy("inv");
  solver_param.set_gamma(0.0001);
  solver_param.set_power(0.75);
  solver_param.set_momentum(0.9);
  solver_param.set_weight_decay(0.0005);
  solver_param.set_type("SGD");

  LOG_IF(INFO, Caffe::root_solver()) << "Init solver...";
  shared_ptr<Solver<double> >
      solver(SolverRegistry<double>::CreateSolver(solver_param));
  solver->Solve(NULL);
  LOG_IF(INFO, Caffe::root_solver())
    << "test end";
#ifdef MYMPI
  MPI_Finalize();
#endif
  return 0;
}
