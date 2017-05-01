#include "caffe/caffe.hpp"
#include "caffe/util/serialize.hpp"
using namespace caffe;

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

int main() {

  DataParameter data_param_data;
  data_param_data.set_source("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
  data_param_data.set_batch_size(100);
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
  data_param_label.set_batch_size(100);
  LayerParameter data_test;
  data_test.set_name("data_test");
  data_test.set_type("Data");
  data_test.add_top("data");
  data_test.add_top("label");
  data_test.setup_data_param(data_param_label);
  NetStateRule test_include;
  test_include.set_phase(TEST);
  data_test.add_include(test_include);

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


  NetParameter net_param;
  net_param.set_name("vgg16");

  net_param.add_layer(data_train);
  net_param.add_layer(data_test);

  net_param.add_layer(vgg_conv(64, "conv1_1_mnist", "data", "conv1_1"));
  net_param.add_layer(vgg_relu("relu1_1", "conv1_1", "conv1_1"));
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

  net_param.add_layer(vgg_ip(4096, "fc6_mnist", "pool5", "fc6"));
  net_param.add_layer(vgg_relu("relu6", "fc6", "fc6"));
  net_param.add_layer(vgg_dropout("drop6", "fc6", "fc6"));
  
  net_param.add_layer(vgg_ip(4096, "fc7_mnist", "fc6", "fc7"));
  net_param.add_layer(vgg_relu("relu7", "fc7", "fc7"));
  net_param.add_layer(vgg_dropout("drop7", "fc7", "fc7"));

  net_param.add_layer(vgg_ip(10, "fc8_mnist", "fc7", "fc8"));
  net_param.add_layer(loss);
  net_param.add_layer(accuracy);

  SolverParameter solver_param;
  solver_param.set_net_param(net_param);
  solver_param.add_test_iter(100);
  solver_param.set_test_interval(500);
  solver_param.set_base_lr(0.01);
  solver_param.set_display(100);
  solver_param.set_max_iter(10000);
  solver_param.set_lr_policy("fixed");
  solver_param.set_momentum(0.9);
  solver_param.set_weight_decay(0.0005);
  solver_param.set_type("SGD");

  Serial_Net net;
  {
    std::ifstream ifs("../data/serialized_caffemodel");
    boost::archive::text_iarchive ia(ifs);
    ia >> net;
  }

  for (int i=0; i<net.layers.size(); i++)
    LOG(INFO) << "layer " << net.layers[i].name;

  shared_ptr<Solver<float> >
      solver(SolverRegistry<float>::CreateSolver(solver_param));
  
  solver->net()->CopyTrainedLayersFrom(net);
  solver->Solve(NULL);

  return 0;
}
