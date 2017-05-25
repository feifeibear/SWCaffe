#include "caffe/caffe.hpp"
#ifdef MYMPI
#include <mpi.h>
#endif
#include <string>
using namespace caffe;

int main (int argc, char ** argv) {
#ifdef MYMPI
  MPI_Init(&argc, &argv);
#endif

  SolverParameter solver_param;
  solver_param.add_test_iter(1);
  solver_param.set_test_interval(50);
  solver_param.set_base_lr(0.01);
  solver_param.set_display(10);
  solver_param.set_max_iter(450000);
  solver_param.set_lr_policy("inv");
  solver_param.set_gamma(0.1);
  solver_param.set_momentum(0.9);
  solver_param.set_weight_decay(0.0005);
  solver_param.set_type("SGD");
  DLOG(INFO) << "Solver Setup OK";


  NetParameter net_param = solver_param.mutable_net_param();
  LayerParameter* tmplp = net_param.add_layer();
  tmplp->set_name("data_train");
  tmplp->set_type("IMAGENETData");
  tmplp->add_top("data");
  tmplp->add_top("label");
  //TODO I double not like write it this way, No mirror crop_size 
  NetStateRule train_include;
  train_include.set_phase(TRAIN);
  tmplp->add_include(train_include);
  DataParameter* data_param_data = tmplp->add_data_param();
  data_param_data->set_source("../data/imagenet_bin/train_data.bin", "../data/imagenet_bin/train_label.bin", "../data/imagenet/train_mean.bin");
  data_param_data->set_batch_size(128);
  DLOG(INFO) << "DataLayer Setup OK";

  tmplp = net_param.add_layer();
  tmplp->set_name("data_test");
  tmplp->set_type("IMAGENETData");
  tmplp->add_top("data");
  tmplp->add_top("label");
  //I double not like write it this way
  NetStateRule test_include;
  test_include.set_phase(TEST);
  tmplp->add_include(test_include);
  DataParameter* data_param_label = tmplp->add_data_param();
  data_param_label->set_source("../data/imagenet_bin/test_data.bin", "../data/imagenet_bin/test_label.bin", "../data/imagenet/test_mean.bin");
  data_param_label->set_batch_size(50);
  DLOG(INFO) << "TEST Setup OK";

  ParamSpec * tmpps;
  ConvolutionParameter* tmpconvp;
  FillerParameter* fillerp;

  //1st conv-lrn-relu-pool layer
  tmplp = net_param.add_layer();
  tmplp->set_name("conv1");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("data");
  tmplp->add_top("conv1");
  DLOG(INFO) << "CONV1 Setup 0.1 OK";
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  DLOG(INFO) << "CONV1 Setup 0.2 OK";
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp = tmplp->add_convolution_param();
  tmpconvp->set_num_output(96);
  DLOG(INFO) << "CONV1 Setup 0.3 OK";
  tmpconvp->add_kernel_size(11);
  DLOG(INFO) << "CONV1 Setup 0.4 OK";
  tmpconvp->add_stride(4);
  DLOG(INFO) << "CONV1 Setup 0.5 OK";
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0);
  DLOG(INFO) << "CONV1 Setup 1 OK";

  tmplp = net_param.add_layer();
  tmplp->set_name("relu1");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv1");
  tmplp->add_top("conv1");
  tmplp->add_relu_param();
  DLOG(INFO) << "CONV1 Setup 2 OK";

  tmplp = net_param.add_layer();
  tmplp->set_name("norm1");
  tmplp->set_type("LRN");
  tmplp->add_bottom("conv1");
  tmplp->add_top("norm1");
  LRNParameter* lrn_param = tmplp->add_lrn_param();
  lrn_param->set_local_size(5);
  lrn_param->set_alpha(0.0001);
  lrn_param->set_beta(0.75);
  DLOG(INFO) << "CONV1 Setup 3 OK";

  PoolingParameter* tmppoolingp;
  tmplp = net_param.add_layer();
  tmplp->set_name("pool1");
  tmplp->set_type("Pooling");
  tmplp->add_bottom("norm1");
  tmplp->add_top("pool1");
  tmppoolingp = tmplp->add_pooling_param();
  tmppoolingp->set_pool(PoolingParameter_PoolMethod_MAX);
  tmppoolingp->set_kernel_size(3);
  tmppoolingp->set_stride(2);
  DLOG(INFO) << "CONV1 Setup 4 OK";


  //2nd conv-lrn-relu-pool layer
  tmplp = net_param.add_layer();
  tmplp->set_name("conv2");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("pool1");
  tmplp->add_top("conv2");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp = tmplp->add_convolution_param();
  tmpconvp->set_num_output(256);
  tmpconvp->add_pad(2);
  tmpconvp->add_kernel_size(5);
  tmpconvp->add_stride(14);
  tmpconvp->set_group(2);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu2");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv2");
  tmplp->add_top("conv2");
  tmplp->add_relu_param();

  //TODO
  tmplp = net_param.add_layer();
  tmplp->set_name("norm2");
  tmplp->set_type("LRN");
  tmplp->add_bottom("conv2");
  tmplp->add_top("norm2");
  lrn_param = tmplp->add_lrn_param();
  lrn_param->set_local_size(5);
  lrn_param->set_alpha(0.0001);
  lrn_param->set_beta(0.75);

  tmplp = net_param.add_layer();
  tmplp->set_name("pool2");
  tmplp->set_type("Pooling");
  tmplp->add_bottom("norm1");
  tmplp->add_top("pool2");
  tmppoolingp = tmplp->add_pooling_param();
  tmppoolingp->set_pool(PoolingParameter_PoolMethod_MAX);
  tmppoolingp->set_kernel_size(3);
  tmppoolingp->set_stride(2);
  DLOG(INFO) << "CONV2 Setup OK";


  //3rd conv-lrn-relu-pool layer
  tmplp = net_param.add_layer();
  tmplp->set_name("conv3");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("pool2");
  tmplp->add_top("conv3");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp = tmplp->add_convolution_param();
  tmpconvp->set_num_output(384);
  tmpconvp->add_pad(1);
  tmpconvp->add_kernel_size(3);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu3");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv3");
  tmplp->add_top("conv3");
  tmplp->add_relu_param();
  DLOG(INFO) << "CONV3 Setup OK";

  //4nd conv
  tmplp = net_param.add_layer();
  tmplp->set_name("conv4");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("conv3");
  tmplp->add_top("conv4");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp = tmplp->add_convolution_param();
  tmpconvp->set_num_output(384);
  tmpconvp->add_pad(1);
  tmpconvp->add_kernel_size(3);
  tmpconvp->set_group(2);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu4");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv4");
  tmplp->add_top("conv4");
  tmplp->add_relu_param();
  DLOG(INFO) << "CONV4 Setup OK";

  //5nd conv + relu
  tmplp = net_param.add_layer();
  tmplp->set_name("conv5");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("conv4");
  tmplp->add_top("conv5");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp = tmplp->add_convolution_param();
  tmpconvp->set_num_output(256);
  tmpconvp->add_pad(1);
  tmpconvp->add_kernel_size(3);
  tmpconvp->set_group(2);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu5");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv5");
  tmplp->add_top("conv5");
  tmplp->add_relu_param();

  tmplp = net_param.add_layer();
  tmplp->set_name("pool5");
  tmplp->set_type("Pooling");
  tmplp->add_bottom("conv5");
  tmplp->add_top("pool5");
  tmppoolingp = tmplp->add_pooling_param();
  tmppoolingp->set_pool(PoolingParameter_PoolMethod_MAX);
  tmppoolingp->set_kernel_size(3);
  tmppoolingp->set_stride(2);
  DLOG(INFO) << "CONV5 Setup OK";

  //layer 6
InnerProductParameter* tmpipp;
  tmplp = net_param.add_layer();
  tmplp->set_name("fc6");
  tmplp->set_type("InnerProduct");
  tmplp->add_bottom("pool5");
  tmplp->add_top("fc6");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpipp = tmplp->add_inner_product_param(); 
  tmpipp->set_num_output(4096);
  fillerp = tmpipp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.005);
  fillerp = tmpipp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu6");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("fc6");
  tmplp->add_top("fc6");
  tmplp->add_relu_param();

  tmplp = net_param.add_layer();
  tmplp->set_name("drop6");
  tmplp->set_type("Dropout");
  tmplp->add_bottom("fc6");
  tmplp->add_top("fc6");
  DropoutParameter* dropp = tmplp->add_dropout_param();
  dropp->set_dropout_ratio(0.5);
  DLOG(INFO) << "CONV6 Setup OK";



  //layer 7
  tmplp = net_param.add_layer();
  tmplp->set_name("fc7");
  tmplp->set_type("InnerProduct");
  tmplp->add_bottom("fc6");
  tmplp->add_top("fc7");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpipp = tmplp->add_inner_product_param(); 
  tmpipp->set_num_output(4096);
  fillerp = tmpipp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.005);
  fillerp = tmpipp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu7");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("fc7");
  tmplp->add_top("fc7");
  tmplp->add_relu_param();

  tmplp = net_param.add_layer();
  tmplp->set_name("drop7");
  tmplp->set_type("Dropout");
  tmplp->add_bottom("fc7");
  tmplp->add_top("fc7");
  dropp = tmplp->add_dropout_param();
  dropp->set_dropout_ratio(0.5);

  //8th layer
  tmplp = net_param.add_layer();
  tmplp->set_name("fc8");
  tmplp->set_type("InnerProduct");
  tmplp->add_bottom("fc7");
  tmplp->add_top("fc8");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpipp = tmplp->add_inner_product_param(); 
  tmpipp->set_num_output(16);
  fillerp = tmpipp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpipp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0);
  DLOG(INFO) << "CONV7 Setup OK";

  //layer 8
  tmplp = net_param.add_layer();
  tmplp->set_name("accuracy");
  tmplp->set_type("Accuracy");
  tmplp->add_bottom("fc8");
  tmplp->add_bottom("label");
  tmplp->add_top("accuracy");
  tmplp->add_include(test_include);

  //layer loss
  tmplp = net_param.add_layer();
  tmplp->set_name("loss");
  tmplp->set_type("SoftmaxWithLoss");
  tmplp->add_bottom("fc8");
  tmplp->add_bottom("label");
  tmplp->add_top("loss");

  solver_param.set_net_param(net_param);
  DLOG(INFO) << "Init solver...";
  shared_ptr<Solver<double> >
      solver(SolverRegistry<double>::CreateSolver(solver_param));
  DLOG(INFO) << "Begin solve...";
  //solver->net()->CopyTrainedLayersFrom("_iter_700.caffemodel");
  solver->Solve(NULL);
  LOG_IF(INFO, Caffe::root_solver())
    << "test end";
#ifdef MYMPI
  MPI_Finalize();
#endif
  return 0;
}
