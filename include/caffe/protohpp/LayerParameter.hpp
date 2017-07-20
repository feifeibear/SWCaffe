#ifndef _LAYERPARAMETER_H_
#define _LAYERPARAMETER_H_
#include <string>
#include <caffe/blob.hpp>
#include <caffe/protohpp/InnerProductParameter.hpp>
#include <caffe/protohpp/InputParameter.hpp>
#include <caffe/protohpp/BlobProto.hpp>
#include <caffe/protohpp/ConvolutionParameter.hpp>
#include <caffe/protohpp/PoolingParameter.hpp>
#include <caffe/protohpp/DataParameter.hpp>
#include <caffe/protohpp/ReLUParameter.hpp>
#include <caffe/protohpp/TransParameter.hpp>
#include <caffe/protohpp/SoftmaxParameter.hpp>
#include <caffe/protohpp/LossParameter.hpp>
#include <caffe/protohpp/AccuracyParameter.hpp>
#include <caffe/protohpp/RecurrentParameter.hpp>
#include <caffe/protohpp/EltwiseParameter.hpp>
#include <caffe/protohpp/ScaleParameter.hpp>
#include <caffe/protohpp/SliceParameter.hpp>
#include <caffe/protohpp/ConcatParameter.hpp>
#include <caffe/protohpp/ReshapeParameter.hpp>
#include <caffe/protohpp/BiasParameter.hpp>
#include <caffe/protohpp/ReductionParameter.hpp>
#include <caffe/protohpp/DropoutParameter.hpp>
#include <caffe/protohpp/DummyDataParameter.hpp>
#include <caffe/protohpp/LRNParameter.hpp>
#include <caffe/protohpp/PowerParameter.hpp>
#include <caffe/common.hpp>

namespace caffe {

enum Phase {
  TRAIN = 0,
  TEST = 1
};
class NetStateRule {
public:
  NetStateRule(){
    has_phase_ = false; 
    has_min_level_ = has_max_level_ = false;
  }

  NetStateRule(const NetStateRule& other){
    this->CopyFrom(other);
  }

  NetStateRule& operator=(const NetStateRule& other){
    this->CopyFrom(other);
    return *this;
  }

  inline void set_phase(Phase phase) { phase_ = phase; has_phase_ = true; }
  inline bool has_phase() const { return has_phase_; }
  inline Phase phase() const { return phase_; }
  inline void set_min_level(int x) { min_level_=x; has_min_level_ = true;}
  inline void set_max_level(int x) { max_level_=x; has_max_level_ = true;}
  inline bool has_min_level() const { return has_min_level_; }
  inline bool has_max_level() const { return has_max_level_; }
  inline int min_level() const { return min_level_; }
  inline int max_level() const { return max_level_; }
  
  inline int stage_size() const { return stage_.size(); }
  inline int not_stage_size() const { return not_stage_.size(); }
  inline std::string stage( int id ) const { return stage_[id]; }
  inline std::string not_stage( int id ) const { return not_stage_[id]; }
  inline std::vector<std::string> stage() const { return stage_; }
  inline std::vector<std::string> not_stage() const { return not_stage_; }
  inline void add_stage(std::string x) { stage_.push_back(x); }
  inline void add_not_stage(std::string x) { not_stage_.push_back(x); }

  void CopyFrom(const NetStateRule& other){
    phase_ = other.phase();
    min_level_ = other.min_level();
    max_level_ = other.max_level();
    stage_ = other.stage();
    not_stage_ = other.not_stage();

    has_phase_ = other.has_phase();
    has_min_level_ = other.has_min_level();
    has_max_level_ = other.has_max_level();
  }

private:
  Phase phase_;
  bool has_phase_;

  int min_level_;
  bool has_min_level_;
  int max_level_;
  bool has_max_level_;

  std::vector<std::string> stage_;
  std::vector<std::string> not_stage_;
};

class ParamSpec {
public:
  enum DimCheckMode {
    STRICT = 0,
    PERMISSIVE = 1
  };

  ParamSpec():has_lr_mult_(true), has_decay_mult_(true),
  lr_mult_(1.0),decay_mult_(1.0) {}

  ParamSpec(const ParamSpec& other){
    this->CopyFrom(other);
  }

  ParamSpec& operator=(const ParamSpec& other){
    this->CopyFrom(other);
    return *this;
  }

  inline float lr_mult() const { return lr_mult_; }
  inline void set_lr_mult(float x) { lr_mult_=x; }
  inline float decay_mult() const { return decay_mult_; }
  inline void set_decay_mult(float x) { decay_mult_=x; }
  inline bool has_lr_mult() const { return has_lr_mult_; }
  inline bool has_decay_mult() const { return has_decay_mult_; }

  inline DimCheckMode share_mode() const { return share_mode_; }
  inline void set_share_mode(DimCheckMode x) { share_mode_=x; }
  inline std::string name() const { return name_; }
  inline void set_name(std::string x) { name_ = x; }

  void CopyFrom(const ParamSpec& other){
    name_ = other.name();
    share_mode_ = other.share_mode();
    lr_mult_ = other.lr_mult();
    decay_mult_ = other.decay_mult();

    has_lr_mult_ = other.has_lr_mult();
    has_decay_mult_ = other.has_decay_mult();
  }

private:
  std::string name_;

  DimCheckMode share_mode_;

  // The multiplier on the global learning rate for this parameter.
  float lr_mult_; // = 1.0;
  bool has_lr_mult_;

  // The multiplier on the global weight decay for this parameter.
  float decay_mult_; // = 1.0;
  bool has_decay_mult_;
};


class LayerParameter {
  public:
    LayerParameter() {
      has_phase_ = false;

      has_input_param_ = false;
      has_inner_product_param_ = false;
      has_convolution_param_ = false;
      has_pooling_param_ = false;
      has_data_param_ = false;
      has_relu_param_ = false;
      has_trans_param_ = false;
      has_softmax_param_ = false;
      has_loss_param_=  false;
      has_accuracy_param_ = false;
      has_recurrent_param_ = false;
      has_eltwise_param_ = false;
      has_scale_param_ = false;
      has_slice_param_ = false;
      has_concat_param_ = false;
      has_reshape_param_ = false;
      has_bias_param_ = false;
      has_reduction_param_ = false;
      has_dropout_param_ = false;
      has_dummy_param_ = false;
      has_lrn_param_ = false;
      has_power_param_ = false;

      data_param_ = NULL;
      input_param_ = NULL;
      inner_product_param_ = NULL;
      convolution_param_ = NULL;
      pooling_param_ = NULL;
      relu_param_ = NULL;
      trans_param_ = NULL;
      softmax_param_ = NULL;
      loss_param_ = NULL;
      accuracy_param_ = NULL;
      recurrent_param_ = NULL;
      eltwise_param_ = NULL;
      scale_param_ = NULL;
      slice_param_ = NULL;
      concat_param_ = NULL;
      reshape_param_ = NULL;
      bias_param_ = NULL;
      reduction_param_ = NULL;
      dropout_param_ = NULL;
      dummy_data_param_ = NULL;
      lrn_param_ = NULL;
      power_param_ = NULL;
    }

    LayerParameter(const LayerParameter& other) {
      data_param_ = NULL;
      input_param_ = NULL;
      inner_product_param_ = NULL;
      convolution_param_ = NULL;
      pooling_param_ = NULL;
      relu_param_ = NULL;
      trans_param_ = NULL;
      softmax_param_ = NULL;
      loss_param_ = NULL;
      accuracy_param_ = NULL;
      recurrent_param_ = NULL;
      eltwise_param_ = NULL;
      scale_param_ = NULL;
      slice_param_ = NULL;
      concat_param_ = NULL;
      reshape_param_ = NULL;
      bias_param_ = NULL;
      reduction_param_ = NULL;
      dropout_param_ = NULL;
      dummy_data_param_ = NULL;
      lrn_param_ = NULL;
      power_param_ = NULL;
      this->CopyFrom(other);
    }

    LayerParameter& operator=(const LayerParameter& other){
      this->CopyFrom(other);
      return *this;
    }

    void CopyFrom(const LayerParameter& other) {
      name_ = other.name();
      type_ = other.type();
      bottom_ = other.bottom();
      top_ = other.top();
      phase_ = other.phase();
      has_phase_ = other.has_phase();
      loss_weight_ = other.loss_weight();
      propagate_down_ = other.propagate_down();

// SAFE_COPY_VEC macro
#define SAFE_COPY_VEC(name)\
    int name##_size = other.name##_size(); \
    name##_.resize(name##_size); \
    for( int i = 0; i < name##_size; ++i ) \
      name##_[i].CopyFrom(other.name(i)); \


      SAFE_COPY_VEC(param);
      SAFE_COPY_VEC(blobs);
      SAFE_COPY_VEC(include);
      SAFE_COPY_VEC(exclude);

      has_input_param_ = other.has_input_param();
      has_inner_product_param_ = other.has_inner_product_param();
      has_convolution_param_ = other.has_convolution_param();
      has_pooling_param_ = other.has_pooling_param();
      has_data_param_ = other.has_data_param();
      has_softmax_param_ = other.has_softmax_param();
      has_relu_param_ = other.has_relu_param();
      has_trans_param_ = other.has_trans_param();
      has_loss_param_ = other.has_loss_param();
      has_accuracy_param_ = other.has_accuracy_param();
      has_recurrent_param_ = other.has_recurrent_param();
      has_eltwise_param_ = other.has_eltwise_param();
      has_scale_param_ = other.has_scale_param();
      has_slice_param_ = other.has_slice_param();
      has_concat_param_ = other.has_concat_param();
      has_reshape_param_ = other.has_reshape_param();
      has_bias_param_ = other.has_bias_param();
      has_reduction_param_ = other.has_reduction_param();
      has_dropout_param_ = other.has_dropout_param();
      has_dummy_param_ = other.has_dummy_param();
      has_lrn_param_ = other.has_lrn_param();
      has_power_param_ = other.has_power_param();

      if(has_input_param_)
        this->mutable_input_param()->CopyFrom(other.input_param());

      if(has_inner_product_param_)
        this->mutable_inner_product_param()->CopyFrom(other.inner_product_param());
      
      if(has_convolution_param_)
        this->mutable_convolution_param()->CopyFrom(other.convolution_param());
      
      if(has_pooling_param_)
        this->mutable_pooling_param()->CopyFrom(other.pooling_param());

      if(has_data_param_)
        this->mutable_data_param()->CopyFrom(other.data_param());
      
      if (has_softmax_param_)
        this->mutable_softmax_param()->CopyFrom(other.softmax_param());
      
      if (has_relu_param_)
        this->mutable_relu_param()->CopyFrom(other.relu_param());

      if (has_trans_param_)
        this->mutable_trans_param()->CopyFrom(other.trans_param());

      if (has_loss_param_)
        this->mutable_loss_param()->CopyFrom(other.loss_param());

      if (has_accuracy_param_)
        this->mutable_accuracy_param()->CopyFrom(other.accuracy_param());

      if (has_recurrent_param_)
        this->mutable_recurrent_param()->CopyFrom(other.recurrent_param());

      if (has_eltwise_param_)
        this->mutable_eltwise_param()->CopyFrom(other.eltwise_param());

      if (has_scale_param_)
        this->mutable_scale_param()->CopyFrom(other.scale_param());

      if (has_slice_param_)
        this->mutable_slice_param()->CopyFrom(other.slice_param());

      if (has_concat_param_)
        this->mutable_concat_param()->CopyFrom(other.concat_param());

      if (has_reshape_param_)
        this->mutable_reshape_param()->CopyFrom(other.reshape_param());

      if (has_bias_param_)
        this->mutable_bias_param()->CopyFrom(other.bias_param());

      if(has_reduction_param_)
        this->mutable_reduction_param()->CopyFrom(other.reduction_param());

      if(has_dropout_param_)
        this->mutable_dropout_param()->CopyFrom(other.dropout_param());

      if(has_dummy_param_)
        this->mutable_dummy_data_param()->CopyFrom(other.dummy_data_param());

      if(has_lrn_param_)
        this->mutable_lrn_param()->CopyFrom(other.lrn_param());

      if(has_power_param_)
        this->mutable_power_param()->CopyFrom(other.power_param());

    }

    void Clear() {
      bottom_.clear();
      top_.clear();
      loss_weight_.clear();
      param_.clear();
      propagate_down_.clear();
      include_.clear();
      exclude_.clear();
      has_input_param_ = false;
      has_inner_product_param_ = false;
      has_convolution_param_ = false;
      has_pooling_param_ = false;
      has_data_param_ = false;
      has_softmax_param_ = false;
      has_relu_param_ = false;
      has_trans_param_ = false;
      has_loss_param_ = false;
      has_accuracy_param_ = false;
      has_recurrent_param_ = false;
      has_eltwise_param_ = false;
      has_scale_param_ = false;
      has_slice_param_ = false;
      has_concat_param_ = false;
      has_reshape_param_ = false;
      has_bias_param_ = false;
      has_reduction_param_ = false;
      has_dropout_param_ = false;
    }

    ~LayerParameter() { 
      if (input_param_ != NULL) delete input_param_;
      if (data_param_ != NULL) delete data_param_;
      if (inner_product_param_ != NULL) delete inner_product_param_;
      if (convolution_param_ != NULL) delete convolution_param_;
      if (pooling_param_ != NULL) delete pooling_param_;
      if (softmax_param_ != NULL) delete softmax_param_;
      if (relu_param_ != NULL) delete relu_param_;
      if (trans_param_ != NULL) delete trans_param_;
      if (loss_param_ != NULL) delete loss_param_;
      if (accuracy_param_ != NULL) delete accuracy_param_;
      if (recurrent_param_ != NULL) delete recurrent_param_;
      if (eltwise_param_ != NULL) delete eltwise_param_;
      if (scale_param_ != NULL) delete scale_param_;
      if (slice_param_ != NULL) delete slice_param_;
      if (concat_param_ != NULL) delete concat_param_;
      if (reshape_param_ != NULL) delete reshape_param_;
      if (bias_param_ != NULL) delete bias_param_;
      if (reduction_param_ != NULL) delete reduction_param_;
      if (dropout_param_ != NULL) delete dropout_param_;
      if (dummy_data_param_ != NULL) delete dummy_data_param_;
      if (lrn_param_ != NULL ) delete lrn_param_; 
      if (power_param_ != NULL) delete power_param_;
    }

    inline const std::string name() const { return name_; }
    void set_name(std::string name) { name_ = name; }

    inline const std::string type() const { return type_; }
    void set_type(std::string type) { type_ = type; }

    inline const std::vector<std::string> bottom() const { return bottom_; }
    inline std::string bottom( int id ) const { return bottom_[id]; }
    inline int bottom_size() const { return bottom_.size(); }
    inline void add_bottom(std::string bottom_name) { bottom_.push_back(bottom_name); /*LOG(INFO)<<"layer "<<name_<<", bottom_size="<<bottom_.size()<<", bottom "<<bottom_name;*/ }
    inline void set_bottom(int id, std::string bottom_name) { bottom_[id] = bottom_name; }

    inline const std::vector<std::string> top() const { return top_; }
    inline std::string top( int id ) const { return top_[id]; }
    inline int top_size() const { return top_.size(); }
    inline void add_top(std::string top_name) { top_.push_back(top_name); /*LOG(INFO)<<"layer "<<name_<<", top_size="<<top_.size()<<", top "<<top_name;*/ }

    inline Phase phase() const {return phase_;}
    inline bool has_phase() const { return has_phase_; }
    inline void set_phase( Phase newphase ) { phase_ = newphase; }

    inline int loss_weight_size() const {return loss_weight_.size();}
    inline float loss_weight(int id) const { return loss_weight_[id]; }
    inline std::vector<float> loss_weight() const { return loss_weight_; }
    inline void add_loss_weight(float loss_weight) { loss_weight_.push_back(loss_weight); }
    inline void clear_loss_weight() { loss_weight_.clear(); }
    
    inline const ParamSpec& param(int i) const { return param_[i]; }
    inline int param_size() const { return param_.size(); }
    inline ParamSpec* add_param() {
      ParamSpec p;
      param_.push_back(p);
      return &param_[param_.size()-1];
    }

    inline const BlobProto& blobs(int i) const { return blobs_[i]; }
    inline int blobs_size() const { return blobs_.size(); }
    
    inline const std::vector<bool> propagate_down() const { return propagate_down_; }
    inline bool propagate_down(int i) const { return propagate_down_[i]; }
    inline int propagate_down_size() const { return propagate_down_.size(); }
    inline void add_propagate_down(bool x) { propagate_down_.push_back(x); }

    inline const NetStateRule& include( int id ) const { return include_[id]; }
    inline int include_size() const { return include_.size(); }
    inline void add_include(const NetStateRule& x) {
      NetStateRule n(x);
      include_.push_back(n);
    }

    inline const NetStateRule& exclude( int id ) const { return exclude_[id]; }
    inline int exclude_size() const { return exclude_.size(); }
    inline void add_exclude(const NetStateRule& x) { 
      NetStateRule n(x);
      exclude_.push_back(n); 
    }

    //for specified layers
    //Input
    void setup_input_param(const InputParameter& other) {
      has_input_param_ = true;
      if (input_param_ == NULL) input_param_ = new InputParameter;
      input_param_->CopyFrom(other);
    }
    InputParameter* mutable_input_param() {
      has_input_param_ = true;
      if (input_param_ == NULL) input_param_ = new InputParameter;
      return input_param_;
    }
    inline const InputParameter& input_param() const { 
      CHECK_NOTNULL(input_param_);
      return *input_param_;
    }
    inline bool has_input_param() const { return has_input_param_; }

    //inner_product
    inline void setup_inner_product_param(const InnerProductParameter& other) {
      has_inner_product_param_ = true;
      if (inner_product_param_ == NULL) inner_product_param_ = new InnerProductParameter;
      inner_product_param_->CopyFrom(other);
    }

    inline InnerProductParameter* add_inner_product_param() {
      has_inner_product_param_ = true;
      if (inner_product_param_ == NULL) inner_product_param_ = new InnerProductParameter;
      return inner_product_param_;
    }

    InnerProductParameter* mutable_inner_product_param() {
      has_inner_product_param_ = true;
      if (inner_product_param_ == NULL) inner_product_param_ = new InnerProductParameter;
      return inner_product_param_;
    }
    inline const InnerProductParameter& inner_product_param() const { 
      CHECK_NOTNULL(inner_product_param_);
      return *inner_product_param_;
    }
    inline bool has_inner_product_param() const { return has_inner_product_param_; }

    //convolution
    inline void setup_convolution_param(const ConvolutionParameter& other) {
      has_convolution_param_ = true;
      if (convolution_param_ == NULL) convolution_param_ = new ConvolutionParameter;
      convolution_param_->CopyFrom(other);
    }
    inline ConvolutionParameter* add_convolution_param() {
      has_convolution_param_ = true;
      if (convolution_param_ == NULL) convolution_param_ = new ConvolutionParameter;
      return convolution_param_;
    }

    inline ConvolutionParameter* mutable_convolution_param() {
      has_convolution_param_ = true;
      if (convolution_param_ == NULL) convolution_param_ = new ConvolutionParameter;
      return convolution_param_;
    }
    inline const ConvolutionParameter& convolution_param() const { 
      CHECK_NOTNULL(convolution_param_);
      return *convolution_param_;
    }
    inline bool has_convolution_param() const { return has_convolution_param_; }

    //pooling
    inline void setup_pooling_param(const PoolingParameter& other) {
      has_pooling_param_ = true;
      if (pooling_param_ == NULL) pooling_param_ = new PoolingParameter;
      pooling_param_->CopyFrom(other);
    }
    inline PoolingParameter* add_pooling_param() {
      has_pooling_param_ = true;
      if (pooling_param_ == NULL) pooling_param_ = new PoolingParameter;
      return pooling_param_;
    }
    inline PoolingParameter* mutable_pooling_param() {
      has_pooling_param_ = true;
      if (pooling_param_ == NULL) pooling_param_ = new PoolingParameter;
      return pooling_param_;
    }
    inline const PoolingParameter& pooling_param() const { 
      CHECK_NOTNULL(pooling_param_);
      return *pooling_param_;
    }
    inline bool has_pooling_param() const { return has_pooling_param_; }

    //data
    void setup_data_param(const DataParameter& other) {
      has_data_param_ = true;
      if (data_param_ == NULL) data_param_ = new DataParameter;
      data_param_->CopyFrom(other);
    }
    inline DataParameter* mutable_data_param() {
      has_data_param_ = true;
      if (data_param_ == NULL) data_param_ = new DataParameter;
      return data_param_;
    }
    inline DataParameter* add_data_param() {
      has_data_param_ = true;
      if (data_param_ == NULL) data_param_ = new DataParameter;
      return data_param_;
    }
    inline const DataParameter& data_param() const { 
      CHECK_NOTNULL(data_param_);
      return *data_param_;
    }
    inline bool has_data_param() const { return has_data_param_; }

    //relu
    inline void setup_relu_param(const ReLUParameter& other) {
      has_relu_param_ = true;
      if (relu_param_ == NULL) relu_param_ = new ReLUParameter;
      relu_param_->CopyFrom(other);
    }
    inline ReLUParameter* add_relu_param() {
      has_relu_param_ = true;
      if (relu_param_ == NULL) relu_param_ = new ReLUParameter;
      return relu_param_;
    }
    ReLUParameter* mutable_relu_param() {
      has_relu_param_ = true;
      if (relu_param_ == NULL) relu_param_ = new ReLUParameter;
      return relu_param_;
    }
    inline const ReLUParameter& relu_param() const {
      CHECK_NOTNULL(relu_param_);
      return *relu_param_;
    }
    inline bool has_relu_param() const { return has_relu_param_; }

    //trans
    inline void setup_trans_param(const TransParameter& other) {
      has_trans_param_ = true;
      if (trans_param_ == NULL) trans_param_ = new TransParameter;
      trans_param_->CopyFrom(other);
    }
    inline TransParameter* add_trans_param() {
      has_trans_param_ = true;
      if (trans_param_ == NULL) trans_param_ = new TransParameter;
      return trans_param_;
    }
    TransParameter* mutable_trans_param() {
      has_trans_param_ = true;
      if (trans_param_ == NULL) trans_param_ = new TransParameter;
      return trans_param_;
    }
    inline const TransParameter& trans_param() const {
      CHECK_NOTNULL(trans_param_);
      return *trans_param_;
    }
    inline bool has_trans_param() const { return has_trans_param_; }



    //Softmax
    void setup_softmax_param(const SoftmaxParameter& other) {
      has_softmax_param_ = true;
      if (softmax_param_ == NULL) softmax_param_ = new SoftmaxParameter;
      softmax_param_->CopyFrom(other);
    }
    SoftmaxParameter* mutable_softmax_param() {
      has_softmax_param_ = true;
      if (softmax_param_ == NULL) softmax_param_ = new SoftmaxParameter;
      return softmax_param_;
    }
    inline const SoftmaxParameter& softmax_param() const {
      return softmax_param_ != NULL ? *softmax_param_ : SoftmaxParameter::default_instance_;
    }
    inline bool has_softmax_param() const { return has_softmax_param_; }

    //Loss
    void setup_loss_param(const LossParameter& other) {
      has_loss_param_ = true;
      if (loss_param_ == NULL) loss_param_ = new LossParameter;
      loss_param_->CopyFrom(other);
    }
    inline LossParameter* add_loss_param() {
      has_loss_param_ = true;
      if (loss_param_ == NULL) loss_param_ = new LossParameter;
      return loss_param_;
    }
    LossParameter* mutable_loss_param() {
      has_loss_param_ = true;
      if (loss_param_ == NULL) loss_param_ = new LossParameter;
      return loss_param_;
    }
    inline const LossParameter& loss_param() const { 
      return loss_param_ != NULL ? *loss_param_ : LossParameter::default_instance_;
    }
    inline bool has_loss_param() const { return has_loss_param_; }

    //Accuracy
    inline AccuracyParameter* add_accuracy_param() {
      has_accuracy_param_ = true;
      if (accuracy_param_ == NULL) accuracy_param_ = new AccuracyParameter;
      return accuracy_param_;
    }
    inline void setup_accuracy_param(const AccuracyParameter& other) {
      has_accuracy_param_ = true;
      if (accuracy_param_ == NULL) accuracy_param_ = new AccuracyParameter;
      accuracy_param_->CopyFrom(other);
    }
    inline AccuracyParameter* mutable_accuracy_param() {
      has_accuracy_param_ = true;
      if (accuracy_param_ == NULL) accuracy_param_ = new AccuracyParameter;
      return accuracy_param_;
    }
    inline const AccuracyParameter& accuracy_param() const { 
      return accuracy_param_ != NULL ? *accuracy_param_ : AccuracyParameter::default_instance_;
    }
    inline bool has_accuracy_param() const { return has_accuracy_param_; }

    //Recurrent
    void setup_recurrent_param(const RecurrentParameter& other) {
      has_recurrent_param_ = true;
      if (recurrent_param_ == NULL) recurrent_param_ = new RecurrentParameter;
      recurrent_param_->CopyFrom(other);
    }
    RecurrentParameter* mutable_recurrent_param() {
      has_recurrent_param_ = true;
      if (recurrent_param_ == NULL) recurrent_param_ = new RecurrentParameter;
      return recurrent_param_;
    }
    inline const RecurrentParameter& recurrent_param() const { 
      CHECK_NOTNULL(recurrent_param_);
      return *recurrent_param_;
    }
    inline bool has_recurrent_param() const { return has_recurrent_param_; }

    //Eltwise
    void setup_eltwise_param(const EltwiseParameter& other) {
      has_eltwise_param_ = true;
      if (eltwise_param_ == NULL) eltwise_param_ = new EltwiseParameter;
      eltwise_param_->CopyFrom(other);
    }
    EltwiseParameter* mutable_eltwise_param() {
      has_eltwise_param_ = true;
      if (eltwise_param_ == NULL) eltwise_param_ = new EltwiseParameter;
      return eltwise_param_;
    }
    inline const EltwiseParameter& eltwise_param() const { 
      CHECK_NOTNULL(eltwise_param_);
      return *eltwise_param_;
    }
    inline bool has_eltwise_param() const { return has_eltwise_param_; }

    //Scale
    void setup_scale_param(const ScaleParameter& other) {
      has_scale_param_ = true;
      if (scale_param_ == NULL) scale_param_ = new ScaleParameter;
      scale_param_->CopyFrom(other);
    }
    ScaleParameter* mutable_scale_param() {
      has_scale_param_ = true;
      if (scale_param_ == NULL) scale_param_ = new ScaleParameter;
      return scale_param_;
    }
    inline const ScaleParameter& scale_param() const { 
      return scale_param_ != NULL ? *scale_param_ : ScaleParameter::default_instance_;
    }
    inline bool has_scale_param() const { return has_scale_param_; }

    //Slice
    void setup_slice_param(const SliceParameter& other) {
      has_slice_param_ = true;
      if (slice_param_ == NULL) slice_param_ = new SliceParameter;
      slice_param_->CopyFrom(other);
    }
    SliceParameter* mutable_slice_param() {
      has_slice_param_ = true;
      if (slice_param_ == NULL) slice_param_ = new SliceParameter;
      return slice_param_;
    }
    inline const SliceParameter& slice_param() const { 
      return slice_param_ != NULL ? *slice_param_ : SliceParameter::default_instance_;
    }
    inline bool has_slice_param() const { return has_slice_param_; }

    //Concat
    void setup_concat_param(const ConcatParameter& other) {
      has_concat_param_ = true;
      if (concat_param_ == NULL) concat_param_ = new ConcatParameter;
      concat_param_->CopyFrom(other);
    }
    ConcatParameter* mutable_concat_param() {
      has_concat_param_ = true;
      if (concat_param_ == NULL) concat_param_ = new ConcatParameter;
      return concat_param_;
    }
    inline const ConcatParameter& concat_param() const { 
      return concat_param_ != NULL ? *concat_param_ : ConcatParameter::default_instance_;
    }
    inline bool has_concat_param() const { return has_concat_param_; }

    //Reshape
    void setup_reshape_param(const ReshapeParameter& other) {
      has_reshape_param_ = true;
      if (reshape_param_ == NULL) reshape_param_ = new ReshapeParameter;
      reshape_param_->CopyFrom(other);
    }
    ReshapeParameter* mutable_reshape_param() {
      has_reshape_param_ = true;
      if (reshape_param_ == NULL) reshape_param_ = new ReshapeParameter;
      return reshape_param_;
    }
    inline const ReshapeParameter& reshape_param() const { 
      CHECK_NOTNULL(reshape_param_);
      return *reshape_param_;
    }
    inline bool has_reshape_param() const { return has_reshape_param_; }

    //Bias
    void setup_bias_param(const BiasParameter& other) {
      has_bias_param_ = true;
      if (bias_param_ == NULL) bias_param_ = new BiasParameter;
      bias_param_->CopyFrom(other);
    }
    BiasParameter* mutable_bias_param() {
      has_bias_param_ = true;
      if (bias_param_ == NULL) bias_param_ = new BiasParameter;
      return bias_param_;
    }
    inline const BiasParameter& bias_param() const { 
      return bias_param_ != NULL ? *bias_param_ : BiasParameter::default_instance_;
    }
    inline bool has_bias_param() const { return has_bias_param_; }

    //Reduction
    void setup_reduction_param(const ReductionParameter& other) {
      has_reduction_param_ = true;
      if (reduction_param_ == NULL) reduction_param_ = new ReductionParameter;
      reduction_param_->CopyFrom(other);
    }
    ReductionParameter* mutable_reduction_param() {
      has_reduction_param_ = true;
      if (reduction_param_ == NULL) reduction_param_ = new ReductionParameter;
      return reduction_param_;
    }
    inline const ReductionParameter& reduction_param() const { 
      return reduction_param_ != NULL ? *reduction_param_ : ReductionParameter::default_instance_;
    }
    inline bool has_reduction_param() const { return has_reduction_param_; }

    //Dropout
    inline DropoutParameter* add_dropout_param() {
      has_dropout_param_ = true;
      if (dropout_param_ == NULL) dropout_param_ = new DropoutParameter;
      return dropout_param_;
    }
    void setup_dropout_param(const DropoutParameter& other) {
      has_dropout_param_ = true;
      if (dropout_param_ == NULL) dropout_param_ = new DropoutParameter;
      dropout_param_->CopyFrom(other);
    }
    DropoutParameter* mutable_dropout_param() {
      has_dropout_param_ = true;
      if (dropout_param_ == NULL) dropout_param_ = new DropoutParameter;
      return dropout_param_;
    }
    inline const DropoutParameter& dropout_param() const { 
      CHECK_NOTNULL(dropout_param_);
      return *dropout_param_;
    }
    inline bool has_dropout_param() const { return has_dropout_param_; }
    //DummyDataParameter
    inline void set_dummy_data_param( const DummyDataParameter& other ) {
      has_dummy_param_ = true;
      if(dummy_data_param_ == NULL) dummy_data_param_ = new DummyDataParameter;
      dummy_data_param_->CopyFrom(other);
    }
    inline DummyDataParameter* mutable_dummy_data_param() {
      has_dummy_param_ = true;
      if(dummy_data_param_ == NULL) dummy_data_param_ = new DummyDataParameter;
      return dummy_data_param_;
    }
    inline const DummyDataParameter& dummy_data_param() const {
      CHECK_NOTNULL(dummy_data_param_);
      return *dummy_data_param_;
    }
    inline DummyDataParameter* add_dummy_data_param() {
      has_dummy_param_ = true;
      if(dummy_data_param_ == NULL) dummy_data_param_ = new DummyDataParameter;
      return dummy_data_param_;
    }
    inline bool has_dummy_param() const { return has_dummy_param_; }


    //LRNParameter
    inline void set_lrn_param( const LRNParameter& other ) {
      has_lrn_param_ = true;
      if(lrn_param_ == NULL) lrn_param_ = new LRNParameter;
      lrn_param_->CopyFrom(other);
    }
    inline LRNParameter* mutable_lrn_param() {
      has_lrn_param_ = true;
      if(lrn_param_ == NULL) lrn_param_ = new LRNParameter;
      return lrn_param_;
    } 
    inline const LRNParameter& lrn_param() const {
      return *lrn_param_;
    }
    inline LRNParameter* add_lrn_param() {
      has_lrn_param_ = true;
      if(lrn_param_ == NULL) lrn_param_ = new LRNParameter;
      return lrn_param_;
    }
    inline bool has_lrn_param() const { return has_lrn_param_; }


    //power
    inline void set_power_param( const PowerParameter& other ) {
      has_power_param_= true;
      if(pooling_param_ == NULL) power_param_ = new PowerParameter;
      power_param_->CopyFrom(other);
    }
    inline PowerParameter* mutable_power_param() {
      has_power_param_ = true;
      if(power_param_ == NULL) power_param_ = new PowerParameter;
      return power_param_;
    }
    inline const PowerParameter& power_param() const {
      CHECK_NOTNULL(power_param_);
      return *power_param_;
    }
    inline PowerParameter* add_power_param() {
      has_power_param_ = true;
      if(power_param_ == NULL) power_param_ = new PowerParameter;
      return power_param_;
    }
    inline bool has_power_param() const { return has_power_param_; }


  private:
    std::string name_;
    std::string type_;
    vector<std::string> bottom_;
    vector<std::string> top_;

    Phase phase_;
    bool has_phase_;

    std::vector<float> loss_weight_;
    std::vector<ParamSpec> param_;
    std::vector<BlobProto> blobs_;

    std::vector<bool> propagate_down_;
    std::vector<NetStateRule> include_;
    std::vector<NetStateRule> exclude_;

  //for BlobProto
  //for specified layers
    InputParameter* input_param_;
    bool has_input_param_;
    InnerProductParameter* inner_product_param_;
    bool has_inner_product_param_;
    ConvolutionParameter* convolution_param_;
    bool has_convolution_param_;
    PoolingParameter* pooling_param_;
    bool has_pooling_param_;
    DataParameter* data_param_;
    bool has_data_param_;
    ReLUParameter* relu_param_;
    bool has_relu_param_;
    TransParameter* trans_param_;
    bool has_trans_param_;
    SoftmaxParameter* softmax_param_;
    bool has_softmax_param_;
    LossParameter* loss_param_;
    bool has_loss_param_;
    AccuracyParameter* accuracy_param_;
    bool has_accuracy_param_;
    RecurrentParameter* recurrent_param_;
    bool has_recurrent_param_;
    EltwiseParameter* eltwise_param_;
    bool has_eltwise_param_;
    ScaleParameter* scale_param_;
    bool has_scale_param_;
    SliceParameter* slice_param_;
    bool has_slice_param_;
    ConcatParameter* concat_param_;
    bool has_concat_param_;
    ReshapeParameter* reshape_param_;
    bool has_reshape_param_;
    BiasParameter* bias_param_;
    bool has_bias_param_;
    ReductionParameter* reduction_param_;
    bool has_reduction_param_;
    DropoutParameter* dropout_param_;
    bool has_dropout_param_;
    DummyDataParameter* dummy_data_param_;
    bool has_dummy_param_;
    LRNParameter* lrn_param_;
    bool has_lrn_param_;
    PowerParameter* power_param_;
    bool has_power_param_;
};

}//end caffe
#endif

