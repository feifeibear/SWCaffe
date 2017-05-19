#include "caffe/caffe.hpp"
using namespace caffe;

int main (int argc, char ** argv) {
#ifdef MYMPI
  MPI_Init(&argc, &argv);
#endif
  DataParameter data_param_data;
  data_param_data.set_source("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", "");
  data_param_data.set_batch_size(100);
  LayerParameter data_train;
  data_train.set_name("data_train");
  data_train.set_type("MNISTData");
  data_train.add_top("data");
  data_train.add_top("clip");
  data_train.add_top("label");
  data_train.setup_data_param(data_param_data);
  NetStateRule train_include;
  train_include.set_phase(TRAIN);
  data_train.add_include(train_include);

  DataParameter data_param_label;
  data_param_label.set_source("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", "");
  data_param_label.set_batch_size(100);
  LayerParameter data_test;
  data_test.set_name("data_test");
  data_test.set_type("MNISTData");
  data_test.add_top("data");
  data_test.add_top("clip");
  data_test.add_top("label");
  data_test.setup_data_param(data_param_label);
  NetStateRule test_include;
  test_include.set_phase(TEST);
  data_test.add_include(test_include);

  RecurrentParameter lstm_param;
  lstm_param.set_num_output(128);
  lstm_param.mutable_weight_filler()->set_type("gaussian");
  lstm_param.mutable_weight_filler()->set_std(0.1);
  lstm_param.mutable_bias_filler()->set_type("constant");
  LayerParameter lstm;
  lstm.set_name("lstm");
  lstm.set_type("LSTM");
  lstm.add_bottom("data");
  lstm.add_bottom("clip");
  lstm.add_top("lstm");
  lstm.setup_recurrent_param(lstm_param);

  SliceParameter slice_param;
  slice_param.set_axis(0);
  slice_param.add_slice_point(27);
  LayerParameter slice;
  slice.set_name("slice");
  slice.set_type("Slice");
  slice.add_bottom("lstm");
  slice.add_top("deprecated_output");
  slice.add_top("sliced_output");
  slice.setup_slice_param(slice_param);

  LayerParameter silence;
  silence.set_name("silence");
  silence.set_type("Silence");
  silence.add_bottom("deprecated_output");

  ReshapeParameter reshape_param;
  reshape_param.mutable_shape()->add_dim(100);
  reshape_param.mutable_shape()->add_dim(128);
  LayerParameter reshape;
  reshape.set_name("reshape");
  reshape.set_type("Reshape");
  reshape.add_bottom("sliced_output");
  reshape.add_top("lstm_last_output");
  reshape.setup_reshape_param(reshape_param);

  InnerProductParameter ip_param1;
  ip_param1.set_num_output(10);
  ip_param1.mutable_weight_filler()->set_type("gaussian");
  ip_param1.mutable_weight_filler()->set_std(0.1);
  ip_param1.mutable_bias_filler()->set_type("constant");
  LayerParameter ip1;
  ip1.set_name("ip1");
  ip1.set_type("InnerProduct");
  ip1.add_bottom("lstm_last_output");
  ip1.add_top("ip1");
  ip1.setup_inner_product_param(ip_param1);

  LossParameter loss_param;
  LayerParameter loss;
  loss.set_name("loss");
  loss.set_type("SoftmaxWithLoss");
  loss.add_bottom("ip1");
  loss.add_bottom("label");
  loss.add_top("loss");
  loss.setup_loss_param(loss_param);

  AccuracyParameter accuracy_param;
  LayerParameter accuracy;
  accuracy.set_name("accuracy");
  accuracy.set_type("Accuracy");
  accuracy.add_bottom("ip1");
  accuracy.add_bottom("label");
  accuracy.add_top("accuracy");
  accuracy.setup_accuracy_param(accuracy_param);
  accuracy.add_include(test_include);

  DLOG(INFO) <<  "layer paramter Initialization is OK!";

  NetParameter net_param;
  net_param.set_name("lstm_mnist");
  net_param.add_layer(data_train);
  net_param.add_layer(data_test);
  net_param.add_layer(lstm);
  net_param.add_layer(slice);
  net_param.add_layer(silence);
  net_param.add_layer(reshape);
  net_param.add_layer(ip1);
  net_param.add_layer(loss);
  net_param.add_layer(accuracy);

  DLOG(INFO) <<  "net paramter Initialization is OK!";

  DLOG(INFO) << "Init solver_param...";
  SolverParameter solver_param;
  DLOG(INFO) << "Set net_param...";
  solver_param.set_net_param(net_param);
  solver_param.add_test_iter(100);
  solver_param.set_test_interval(500);
  solver_param.set_base_lr(0.001);
  solver_param.set_display(100);
  solver_param.set_max_iter(10000);
  solver_param.set_lr_policy("fixed");
  solver_param.set_type("Adam");

  DLOG(INFO) << "Init solver...";
  shared_ptr<Solver<float> >
      solver(SolverRegistry<float>::CreateSolver(solver_param));
  DLOG(INFO) << "Begin solve...";
  solver->Solve(NULL);

  DLOG(INFO) << "test end";

#ifdef MYMPI
  MPI_Finalize();
#endif
  return 0;
}
