#include <cstdio>

#include <string>
#include <vector>
#include <sys/time.h>
#ifdef SWMPI
#include <mpi.h> // added by zcj
extern "C"
{
#include "caffe/util/sw_dnn.h"
}
#include "caffe/util/mpi.hpp"
#include "caffe/util/math_functions.hpp"
#endif

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
//#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/collectives.h"
namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_(), callbacks_(), requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_(), callbacks_(), requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
#ifndef SWMPI
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
#else
  LOG(INFO) << "rank " << Caffe::mpi_rank() << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
#endif
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  if (Caffe::root_solver()) {
    LOG(INFO) << "Solver scaffolding done.";
  }

  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
#ifndef SWMPI
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
#else
    LOG(INFO) << "rank " << Caffe::mpi_rank()
        << "Creating training net from net file: " << param_.net();
#endif
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  net_.reset(new Net<Dtype>(net_param));
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

  struct timeval ts, te;
  double calc_lapse = 0.0, comm_lapse = 0.0, tmp_comm_lapse = 0.0;

#ifdef SWMPI
// modified by zcj 2018/10/29
  double t_start,t_copy1,t_copy,t_allreduce,t_applyupdate;
  t_start = MPI_Wtime();
  int param_size =0;
  std::vector<shared_ptr<Blob<Dtype> > > net_params = this->net_->params();
  LOG(INFO) << "net_params size :" << net_params.size();
  for(int i = 0; i < net_params.size(); i++)
  {
     Blob<Dtype>* net_param = net_params[i].get();
     param_size += net_param -> count();
  }
  LOG(INFO) << "param_size :" << param_size << std::endl;
// zhuchuanjia reallocate buffer for weight
  float *local_diff = (float *)malloc(param_size*sizeof(float));
  int offset = 0;
  for(int i =0;i < net_params.size(); i ++)
  {
    Blob<Dtype>* net_param = net_params[i].get();
    sw_memcpy_f((float *)net_param->mutable_cpu_diff(),local_diff+offset, net_param->count());
    net_param->set_cpu_diff(local_diff + offset);
    offset += net_param->count();
  }
// zhuchuanjia bcast W before forward & backward
  for(int i=0; i<this->net_->layers().size(); i++){
        for(int nblobs = 0; nblobs < this->net_->layers()[i]->blobs().size(); nblobs++){
          caffe_mpi_bcast<Dtype>(
              this->net_->layers()[i]->blobs()[nblobs]->mutable_cpu_data(),
              this->net_->layers()[i]->blobs()[nblobs]->count(),
              0,
              MPI_COMM_WORLD);
      }
  }
  t_copy1 = MPI_Wtime() - t_start;
#endif

  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();

    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      if (Caffe::root_solver()) {
#ifdef DEBUG_VERBOSE_1
    gettimeofday(&ts, NULL);
#endif
    TestAll();
#ifdef DEBUG_VERBOSE_1
    gettimeofday(&te, NULL);
    tmp_comm_lapse = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
    LOG(INFO) << "Root: Test cost time is " << tmp_comm_lapse << "sec";
#endif
      }
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
#ifdef DEBUG_VERBOSE_1
    gettimeofday(&ts, NULL);
#endif
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
        loss += net_->ForwardBackward();
    }
#ifdef DEBUG_VERBOSE_1
    gettimeofday(&te, NULL);
    tmp_comm_lapse = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
    calc_lapse += tmp_comm_lapse;
#ifdef SWMPI 
    LOG(INFO) <<" rank " << Caffe::mpi_rank() << "Root: net ForwardBackward time is " << tmp_comm_lapse << "sec";
#else
    LOG_IF(INFO, Caffe::root_solver()) << "Root: net ForwardBackward time is " << tmp_comm_lapse << "sec";
#endif
#endif

#ifdef SWMPI
      loss /= param_.iter_size();
      // average the loss across iterations for smoothed reporting
      UpdateSmoothedLoss(loss, start_iter, average_loss);
      if (display) {
        float lapse = iteration_timer_.Seconds();
        float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
        LOG(INFO) << "rank " << Caffe::mpi_rank() << "MPIRoot: Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
        iteration_timer_.Start();
        iterations_last_ = iter_;
        const vector<Blob<Dtype>*>& result = net_->output_blobs();
        int score_index = 0;
        for (int j = 0; j < result.size(); ++j) {
          const Dtype* result_vec = result[j]->cpu_data();
          const string& output_name =
              net_->blob_names()[net_->output_blob_indices()[j]];
          const Dtype loss_weight =
              net_->blob_loss_weights()[net_->output_blob_indices()[j]];
          for (int k = 0; k < result[j]->count(); ++k) {
            ostringstream loss_msg_stream;
            if (loss_weight) {
              loss_msg_stream << " (* " << loss_weight
                              << " = " << loss_weight * result_vec[k] << " loss)";
            }
            LOG(INFO) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
          }
        }
      }

      for (int i = 0; i < callbacks_.size(); ++i) {
        callbacks_[i]->on_gradients_ready();
      }
      // modified by zcj
      t_start = MPI_Wtime();
      t_copy = 0.0;
      //memset(local_diff, 0, sizeof(float)*param_size);
      //memset(global_diff,0, sizeof(float)*param_size);
      //int offset = 0;
      //for(int i =0;i < net_params.size(); i ++)
      //{
        //Blob<Dtype>* net_param = net_params[i].get();
        //sw_memcpy_f((float *)net_param->mutable_cpu_diff(),local_diff+offset, net_param->count());
        //offset += net_param->count();
      //}
      //for(int i=0;i<param_size;i++)
      //{
        //local_diff[i] /= (float)Caffe::mpi_count();
        //local_diff[i] /= 1;
      //}
      caffe_scal<Dtype>(param_size,1./Caffe::mpi_count(),(Dtype *)local_diff);

      t_copy += MPI_Wtime() - t_start;

      t_start = MPI_Wtime();
      //MPI_Allreduce(
          //local_diff,
          //global_diff,
          //param_size,
          //MPI_FLOAT,
          //MPI_SUM,
          //MPI_COMM_WORLD
          //);
      RingAllreduce(local_diff,param_size,-1);
      //TreeAllreduce(local_diff,param_size,-1);
      t_allreduce = MPI_Wtime() - t_start;

      //for(int i=0;i<param_size;i++)
      //{
         //printf("%f\n",local_diff[i]);
      //}

      //t_start = MPI_Wtime();
      //offset = 0;
      //for(int i =0;i < net_params.size(); i ++)
      //{
        //Blob<Dtype>* net_param = net_params[i].get();
        //sw_memcpy_f(local_diff+offset,(float *)net_param->mutable_cpu_diff(), net_param->count());
        //offset += net_param->count();
      //}
      //free(local_diff);
      //free(global_diff);
      //t_copy += MPI_Wtime() - t_start;
      t_start = MPI_Wtime();
      ApplyUpdate();
      t_applyupdate = MPI_Wtime() - t_start;

      //LOG(INFO) << " t_copy time  " << t_copy + t_copy1
                //<< " extra Initialize time " << t_copy1
                //<< " copy time " << t_copy
                //<< " t_allreduce time " << t_allreduce
                //<< " t_applyupdate time " << t_applyupdate
                //<< std::endl;
    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

      // Save a snapshot if needed.
      if ((param_.snapshot()
           && iter_ % param_.snapshot() == 0) ||
         (request == SolverAction::SNAPSHOT)) {
        Snapshot();
      }

    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }//while
  free(local_diff);
#else
    loss /= param_.iter_size();
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }//while
#endif

} //step

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
#ifdef SWMPI
  if (Caffe::mpi_root_solver()) {
    if (param_.snapshot_after_train()
        && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
      Snapshot();
    }
    if (requested_early_exit_) {
      LOG(INFO) << "MPIRoot: Optimization stopped early.";
      return;
    }
  }
#else
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
#endif
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
#ifdef SWMPI
#ifndef SWMPITEST
  if (Caffe::mpi_root_solver()) {
#else
  if (!Caffe::mpi_root_solver()){
#endif
    if (param_.display() && iter_ % param_.display() == 0) {
      int average_loss = this->param_.average_loss();
      Dtype loss;
      net_->Forward(&loss); 
      UpdateSmoothedLoss(loss, start_iter, average_loss);
      LOG(INFO) << "MPIRoot: Iteration " << iter_ << ", loss = " << smoothed_loss_;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
      TestAll();
    }
    LOG(INFO) << "MPIRoot: Optimization Done.";
  }
#else
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
#endif
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
#ifndef SWMPITEST
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
#else
  LOG_IF(INFO, Caffe::mpi_rank()==1) << "Rank "<< Caffe::mpi_rank() 
    <<" : Iteration " << iter_
    << ", Testing net (#" << test_net_id << ")";
#endif
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
#ifndef SWMPITEST
    LOG(INFO) << "Test loss: " << loss;
#else
    LOG(INFO) << "Rank "<<Caffe::mpi_rank()
      <<" : Test loss: "<<loss;
#endif
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
#ifndef SWMPITEST
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
#else
    LOG(INFO) <<"Rank "<<Caffe::mpi_rank()
      << "    Test net output #" << i << ": " << output_name << " = "
      << mean_score << loss_msg_stream.str();
#endif
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
#ifdef SWMPI
  CHECK(Caffe::mpi_root_solver());
#else
  CHECK(Caffe::root_solver());
#endif
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
