#ifndef _SOLVERPARAMETER_H_
#define _SOLVERPARAMETER_H_

#include "caffe/protohpp/NetParameter.hpp"
#include <string>
#include <vector>

namespace caffe {

enum SolverParameter_SnapshotFormat {
  SolverParameter_SnapshotFormat_HDF5 = 0,
  SolverParameter_SnapshotFormat_BINARYPROTO = 1
};

enum SolverParameter_SolverMode {
  SolverParameter_SolverMode_CPU = 0,
  SolverParameter_SolverMode_GPU = 1
};

class SolverParameter {
public:
	SolverParameter(){}
	explicit SolverParameter(int test_iter, int test_interval, float base_lr, int display,
      int max_iter, std::string lr_policy, float momentum, float weight_decay,
      std::string type="SGD", int num_test=1
			):
    test_interval_(test_interval), base_lr_(base_lr), display_(display),
    max_iter_(max_iter), lr_policy_(lr_policy), momentum_(momentum), weight_decay_(weight_decay),
    type_(type), num_test_(num_test)
	{
		has_net_ = has_net_param_ = has_train_net_ = has_train_net_param_ = false;
    test_compute_loss_ = false;
    test_initialization_ = true;
    average_loss_ = 1;
    iter_size_ = 1;
    regularization_type_ = "L2";
    clip_gradients_ = -1;
    snapshot_ = 0;
    snapshot_diff_ = false;
    snapshot_format_ = SolverParameter_SnapshotFormat_HDF5;
    solver_mode_ = SolverParameter_SolverMode_CPU;
    device_id_ = 0;
    random_seed_ = -1;
    debug_info_ = false;
    snapshot_after_train_ = true;
    layer_wise_reduce_ = true;

    test_iter_.resize(num_test_);
    for (int i=0;i<num_test_; i++) 
      test_iter_[i] = test_iter;
	}

	inline void set_train_net(const NetParameter& train_net_param) { 
		train_net_param_.CopyFrom(train_net_param);
		has_train_net_param_ = true;
	}
	inline void set_test_net(const std::vector<NetParameter>& test_net_param) { 
		test_net_param_.resize(num_test_);
    for (int i=0; i<num_test_; i++) 
      test_net_param_[i].CopyFrom(test_net_param[i]);
	}
	inline void set_net(const NetParameter& net_param) { 
		net_param_.CopyFrom(net_param);
		has_net_param_ = true;
	}

  void CopyFrom(const SolverParameter& other) {
    num_test_ = other.num_test();
    net_param_.CopyFrom(other.net_param());
    train_net_param_.CopyFrom(other.train_net_param());
    if (other.test_net_param_size() > 0) {
      test_net_param_.resize(num_test_);
      for (int i=0; i<num_test_; i++) 
        test_net_param_[i].CopyFrom(other.test_net_param(i));
    }

    test_iter_ = other.test_iter_vec();
    test_interval_ = other.test_interval();
    test_compute_loss_ = other.test_compute_loss();
    test_initialization_ = other.test_initialization();
    base_lr_ = other.base_lr();
    display_ = other.display();
    average_loss_ = other.average_loss();
    max_iter_ = other.max_iter();
    iter_size_ = other.iter_size();
    lr_policy_ = other.lr_policy();
    gamma_ = other.gamma();
    power_ = other.power();
    momentum_ = other.momentum();
    weight_decay_ = other.weight_decay();
    regularization_type_ = other.regularization_type();
    stepsize_ = other.stepsize();
    stepvalue_ = other.stepvalue_vec();
    clip_gradients_ = other.clip_gradients();
    snapshot_ = other.snapshot();
    snapshot_prefix_ = snapshot_prefix();
    snapshot_diff_ = other.snapshot_diff();
    snapshot_format_ = other.snapshot_format();
    solver_mode_ = other.solver_mode();
    device_id_ = other.device_id();
    random_seed_ = other.random_seed();
    type_ = other.type();
    delta_ = other.delta();
    momentum2_ = other.momentum2();
    rms_decay_ = other.rms_decay();
    debug_info_ = other.debug_info();
    snapshot_after_train_ = other.snapshot_after_train();
    layer_wise_reduce_ = other.layer_wise_reduce();
  }

  inline int num_test() const { return num_test_; }
  inline const NetParameter& net_param() const { return net_param_; }
  inline const NetParameter& train_net_param() const { return train_net_param_; }
  inline const NetParameter& test_net_param(int i) const { return test_net_param_[i]; }
  inline const std::vector<NetParameter>& test_net_param_vec() const { return test_net_param_; }
  inline int test_iter(int i) const { return test_iter_[i]; }
  inline int test_iter_size() const { return test_iter_.size(); }
  inline std::vector<int> test_iter_vec() const { return test_iter_; }
  inline int test_interval() const { return test_interval_; }
  inline bool test_compute_loss() const { return test_compute_loss_; }
  inline bool test_initialization() const { return test_initialization_; }
  inline float base_lr() const { return base_lr_; }
  inline int display() const { return display_; }
  inline int average_loss() const { return average_loss_; }
  inline int max_iter() const { return max_iter_; }
  inline int iter_size() const { return iter_size_; }
  inline std::string lr_policy() const { return lr_policy_; }
  inline float gamma() const { return gamma_; }
  inline float power() const { return power_; }
  inline float momentum() const { return momentum_; }
  inline float weight_decay() const { return weight_decay_; }
  inline std::string regularization_type() const { return regularization_type_; }
  inline int stepsize() const { return stepsize_; }
  inline int stepvalue(int i) const { return stepvalue_[i]; }
  inline int stepvalue_size() const { return stepvalue_.size(); }
  inline std::vector<int> stepvalue_vec() const { return stepvalue_; }
  inline float clip_gradients() const { return clip_gradients_; }
  inline int snapshot() const { return snapshot_; }
  inline std::string snapshot_prefix() const { return snapshot_prefix_; }
  inline bool snapshot_diff() const { return snapshot_diff_; }
  inline SolverParameter_SnapshotFormat snapshot_format() const { return snapshot_format_; }
  inline SolverParameter_SolverMode solver_mode() const { return solver_mode_; }
  inline int device_id() const { return device_id_; }
  inline int random_seed() const { return random_seed_; }
  inline std::string type() const { return type_; }
  inline float delta() const { return delta_; }
  inline float momentum2() const { return momentum2_; }
  inline float rms_decay() const { return rms_decay_; }
  inline bool debug_info() const { return debug_info_; }
  inline bool snapshot_after_train() const { return snapshot_after_train_; }
  inline bool layer_wise_reduce() const { return layer_wise_reduce_; }
  inline int test_state_size() const { return test_state_.size(); }

  inline bool has_net() const { return has_net_; }
  inline bool has_train_net() const { return has_train_net_; }
  inline bool has_net_param() const { return has_net_param_; }
  inline bool has_train_net_param() const { return has_train_net_param_; }
  inline bool has_snapshot_prefix() const { return has_snapshot_prefix_; }
  inline int test_net_size() const { return test_net_.size(); }
  inline int test_net_param_size() const { return test_net_param_.size(); }

private:
  int num_test_;

	// DEPRECATED
	std::string net_, train_net_;
	std::vector<std::string> test_net_;

	NetParameter net_param_, train_net_param_;
	std::vector<NetParameter> test_net_param_;

	// DEPRECATED
	NetState train_state_;
	std::vector<NetState> test_state_;
	
	std::vector<int> test_iter_;
	int test_interval_;
	bool test_compute_loss_;
	bool test_initialization_;
  float base_lr_;
  int display_;
  int average_loss_;
  int max_iter_;
  int iter_size_;

  std::string lr_policy_;
  float gamma_, power_;
  float momentum_, weight_decay_;
  std::string regularization_type_;
  int stepsize_;
  std::vector<int> stepvalue_;
  float clip_gradients_;
  int snapshot_;
  std::string snapshot_prefix_;
  bool snapshot_diff_;
  SolverParameter_SnapshotFormat snapshot_format_;
  SolverParameter_SolverMode solver_mode_;
  int device_id_;
  int random_seed_;
  std::string type_;
  float delta_, momentum2_, rms_decay_;
  bool debug_info_, snapshot_after_train_, layer_wise_reduce_;

	bool has_net_, has_train_net_;
	bool has_net_param_, has_train_net_param_;
  bool has_snapshot_prefix_;
};

}

#endif