#include "caffe/caffe.hpp"
#include <mpi.h>
#include "caffe/util/serialize.hpp"
using namespace caffe;

#define net_param_add_layer_vgg_conv(num_output, name, bottom, top) \
  { \
    ConvolutionParameter conv_param; \
    conv_param.set_num_output(num_output); \
    conv_param.set_kernel_h(3); \
    conv_param.set_kernel_w(3); \
    conv_param.set_pad_h(1); \
    conv_param.set_pad_w(1); \
    LayerParameter conv; \
    conv.set_name(name); \
    conv.set_type("Convolution"); \
    conv.add_bottom(bottom); \
    conv.add_top(top); \
    conv.setup_convolution_param(conv_param); \
    net_param.add_layer(conv); \
  }

LayerParameter vgg_conv(int num_output, std::string name, std::string bottom, std::string top) {
  ConvolutionParameter conv_param;
  conv_param.set_num_output(num_output);
  conv_param.set_pad_h(1);
  conv_param.set_pad_w(1);
  conv_param.set_kernel_h(3);
  conv_param.set_kernel_w(3);
  LayerParameter conv;
  conv.set_name(name);
  conv.set_type("Convolution");
  conv.add_bottom(bottom);
  conv.add_top(top);
  conv.setup_convolution_param(conv_param);
  return conv;
}

LayerParameter vgg_pool(std::string name, std::string bottom, std::string top) {
  PoolingParameter pool_param;
  pool_param.set_kernel_size(2);
  pool_param.set_stride(2);
  LayerParameter pool;
  pool.set_name(name);
  pool.set_type("Pooling");
  pool.add_bottom(bottom);
  pool.add_top(top);
  pool.setup_pooling_param(pool_param);
  return pool;
}

LayerParameter vgg_relu(std::string name, std::string bottom, std::string top) {
  ReLUParameter relu_param;
  LayerParameter relu;
  relu.set_name(name);
  relu.set_type("ReLU");
  relu.add_bottom(bottom);
  relu.add_top(top);
  relu.setup_relu_param(relu_param);
  return relu;
}

LayerParameter vgg_ip(int num_output, std::string name, std::string bottom, std::string top) {
  InnerProductParameter ip_param;
  ip_param.set_num_output(num_output);
  LayerParameter ip;
  ip.set_name(name);
  ip.set_type("InnerProduct");
  ip.add_bottom(bottom);
  ip.add_top(top);
  ip.setup_inner_product_param(ip_param);
  return ip;
}

LayerParameter vgg_ip_f8(int num_output, std::string name, std::string bottom, std::string top) {
  InnerProductParameter ip_param;
  ip_param.set_num_output(num_output);
  LayerParameter ip;
  ip.set_name(name);
  ip.set_type("InnerProduct");
  ip.add_bottom(bottom);
  ip.add_top(top);
  ip.setup_inner_product_param(ip_param);
  ParamSpec* ps1 = ip.add_param();
  ps1->set_lr_mult(10);
  ParamSpec* ps2 = ip.add_param();
  ps2->set_lr_mult(10);
  return ip;
}


LayerParameter vgg_dropout(std::string name, std::string bottom, std::string top) {
  DropoutParameter dropout_param;
  LayerParameter dropout;
  dropout.set_name(name);
  dropout.set_type("Dropout");
  dropout.add_bottom(bottom);
  dropout.add_top(top);
  dropout.setup_dropout_param(dropout_param);
  return dropout;
}

int main (int argc, char ** argv) {
#ifdef MYMPI
  MPI_Init(&argc, &argv);
#ifdef DEBUG_VERBOSE_1
//google::InitGoogleLogging((const char *)argv[0]);
//google::SetLogDestination(google::GLOG_INFO,"./log/");
#endif
#endif
  Caffe::set_random_seed(1);

  DataParameter data_param_data;
  data_param_data.set_source("../data/imagenet_bin/train_data.bin", "../data/imagenet_bin/train_label.bin", "../data/imagenet/train_mean.bin");
  data_param_data.set_batch_size(128);
  LayerParameter data_train;
  data_train.set_name("data_train");
  data_train.set_type("IMAGENETData");
  data_train.add_top("data");
  data_train.add_top("label");
  data_train.setup_data_param(data_param_data);
  NetStateRule train_include;
  train_include.set_phase(TRAIN);
  data_train.add_include(train_include);

#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : " << "Train_Data param set done!";
#endif

  DataParameter data_param_label;
  data_param_label.set_source("../data/imagenet_bin/test_data.bin", "../data/imagenet_bin/test_label.bin", "../data/imagenet/test_mean.bin");
  data_param_label.set_batch_size(1);
  LayerParameter data_test;
  data_test.set_name("data_test");
  data_test.set_type("IMAGENETData");
  data_test.add_top("data");
  data_test.add_top("label");
  data_test.setup_data_param(data_param_label);
  NetStateRule test_include;
  test_include.set_phase(TEST);
  data_test.add_include(test_include);

#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : " << "Test_Data param set done!";
#endif

  LossParameter loss_param;
  LayerParameter loss;
  loss.set_name("loss");
  loss.set_type("SoftmaxWithLoss");
  loss.add_bottom("fc8");
  loss.add_bottom("label");
  loss.add_top("loss");
  loss.setup_loss_param(loss_param);

  AccuracyParameter accuracy_param;
  LayerParameter accuracy;
  accuracy.set_name("accuracy");
  accuracy.set_type("Accuracy");
  accuracy.add_bottom("fc8");
  accuracy.add_bottom("label");
  accuracy.add_top("accuracy");
  accuracy.setup_accuracy_param(accuracy_param);
  accuracy.add_include(test_include);

#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : " << "Loss and Accuracy param set done!";
#endif

  NetParameter net_param;
  net_param.set_name("vgg16");

  net_param.add_layer(data_train);
  net_param.add_layer(data_test);
  net_param.add_layer(vgg_conv(64, "conv1_1", "data", "conv1_1"));
  net_param.add_layer(vgg_relu("relu1_1", "conv1_1", "conv1_1"));

  TransParameter trans_param(128, 3, 224, 224, true);
  LayerParameter trans;
  trans.set_name("trans_1");
  trans.set_type("Trans");
  trans.add_bottom("conv1_1");
  trans.add_top("conv1_1");
  trans.setup_trans_param(trans_param);
  net_param.add_layer(trans);

  net_param.add_layer(vgg_conv(64, "conv1_2", "conv1_1", "conv1_2"));
  net_param.add_layer(vgg_relu("relu1_2", "conv1_2", "conv1_2"));
  net_param.add_layer(vgg_pool("pool1", "conv1_2", "pool1"));
  net_param.add_layer(vgg_conv(128, "conv2_1", "pool1", "conv2_1"));
  net_param.add_layer(vgg_relu("relu2_1", "conv2_1", "conv2_1"));
  net_param.add_layer(vgg_conv(128, "conv2_2", "conv2_1", "conv2_2"));
  net_param.add_layer(vgg_relu("relu2_2", "conv2_2", "conv2_2"));
  net_param.add_layer(vgg_pool("pool2", "conv2_2", "pool2"));

  net_param.add_layer(vgg_conv(256, "conv3_1", "pool2", "conv3_1"));
  net_param.add_layer(vgg_relu("relu3_1", "conv3_1", "conv3_1"));
  net_param.add_layer(vgg_conv(256, "conv3_2", "conv3_1", "conv3_2"));
  net_param.add_layer(vgg_relu("relu3_2", "conv3_2", "conv3_2"));
  net_param.add_layer(vgg_conv(256, "conv3_3", "conv3_2", "conv3_3"));
  net_param.add_layer(vgg_relu("relu3_3", "conv3_3", "conv3_3"));
  net_param.add_layer(vgg_pool("pool3", "conv3_3", "pool3"));

  net_param.add_layer(vgg_conv(512, "conv4_1", "pool3", "conv4_1"));
  net_param.add_layer(vgg_relu("relu4_1", "conv4_1", "conv4_1"));
  net_param.add_layer(vgg_conv(512, "conv4_2", "conv4_1", "conv4_2"));
  net_param.add_layer(vgg_relu("relu4_2", "conv4_2", "conv4_2"));
  net_param.add_layer(vgg_conv(512, "conv4_3", "conv4_2", "conv4_3"));
  net_param.add_layer(vgg_relu("relu4_3", "conv4_3", "conv4_3"));
  net_param.add_layer(vgg_pool("pool4", "conv4_3", "pool4"));

  net_param.add_layer(vgg_conv(512, "conv5_1", "pool4", "conv5_1"));
  net_param.add_layer(vgg_relu("relu5_1", "conv5_1", "conv5_1"));
  net_param.add_layer(vgg_conv(512, "conv5_2", "conv5_1", "conv5_2"));
  net_param.add_layer(vgg_relu("relu5_2", "conv5_2", "conv5_2"));
  net_param.add_layer(vgg_conv(512, "conv5_3", "conv5_2", "conv5_3"));
  net_param.add_layer(vgg_relu("relu5_3", "conv5_3", "conv5_3"));
  net_param.add_layer(vgg_pool("pool5", "conv5_3", "pool5"));

  TransParameter trans_param2(128, 512, 7, 7, false);
  LayerParameter trans2;
  trans2.set_name("trans_2");
  trans2.set_type("Trans");
  trans2.add_bottom("pool5");
  trans2.add_top("pool5");
  trans2.setup_trans_param(trans_param2);
  net_param.add_layer(trans2);

  net_param.add_layer(vgg_ip(4096, "fc6", "pool5", "fc6"));
  net_param.add_layer(vgg_relu("relu6", "fc6", "fc6"));
  net_param.add_layer(vgg_dropout("drop6", "fc6", "fc6"));

  net_param.add_layer(vgg_ip(4096, "fc7", "fc6", "fc7"));
  net_param.add_layer(vgg_relu("relu7", "fc7", "fc7"));
  net_param.add_layer(vgg_dropout("drop7", "fc7", "fc7"));

  net_param.add_layer(vgg_ip_f8(16, "fc8_mnist", "fc7", "fc8"));
  net_param.add_layer(loss);
  net_param.add_layer(accuracy);

#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : " << "Net param set done!";
#endif

  SolverParameter solver_param;
  solver_param.set_net_param(net_param);
  solver_param.add_test_iter(1);
  solver_param.set_test_interval(500);
  solver_param.set_base_lr(0.0002);
  solver_param.set_display(10);
  solver_param.set_max_iter(10000);
  solver_param.set_lr_policy("fixed");
  solver_param.set_momentum(0.9);
  solver_param.set_weight_decay(0.0005);
  solver_param.set_type("SGD");

#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : Solver param set done!";
#endif

  Serial_Net net;
  if(Caffe::root_solver()) {
    std::ifstream ifs("../data/serialized_caffemodel");
    net.serialize_in(ifs);
    ifs.close();
    for (int i=0; i<net.layers.size(); i++)
      LOG(INFO) << "Root: layer " << net.layers[i].name;
  }

#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : Read in model done!";
#ifdef MYMPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

  //shared_ptr<Solver<float> >
  //    solver(SolverRegistry<float>::CreateSolver(solver_param));

  shared_ptr<Solver<double> >
      solver(SolverRegistry<double>::CreateSolver(solver_param));
#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : init solver done!";
#endif
  //solver->net()->CopyTrainedLayersFrom(net);
#ifdef DEBUG_VERBOSE_1
  LOG(INFO) << "Rank " << Caffe::solver_rank() << " : Readin Net done!";
#endif
  solver->Solve(NULL);
#ifdef MYMPI
  MPI_Finalize();
#endif

  return 0;
}
