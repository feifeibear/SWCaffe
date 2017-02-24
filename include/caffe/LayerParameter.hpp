#ifndef _LAYERPARAMETER_H_
#define _LAYERPARAMETER_H_
#include <string>
#include <caffe/blob.hpp>
#include <caffe/protohpp/InnerProductParameter.hpp>
#include <caffe/protohpp/InputParameter.hpp>
//#include <caffe/common.hpp>

namespace caffe {

enum Phase {
  TRAIN = 0,
  TEST = 1
};
class NetStateRule {
  // Set phase to require the NetState have a particular phase (TRAIN or TEST)
  // to meet this rule.
public:
  //TODO
  NetStateRule(){
   has_phase_ = false; 
  }
  ~NetStateRule() {}
  bool has_phase() const { return has_phase_; }
  Phase phase() const { return phase_; }
  bool has_min_level() const { return has_min_level_; }
  bool has_max_level() const { return has_max_level_; }
  int min_level() const { return min_level_; }
  int max_level() const { return max_level_; }
  int stage_size() const { return stage_.size(); }
  int not_stage_size() const { return not_stage_.size(); }
  std::string stage( int id ) const { return stage_[id]; }
  std::string not_stage( int id ) const { return not_stage_[id]; }
private:
  Phase phase_;
  bool has_phase_;

  // Set the minimum and/or maximum levels in which the layer should be used.
  // Leave undefined to meet the rule regardless of level.
  int min_level_;
  bool has_min_level_;
  int max_level_;
  bool has_max_level_;

  // Customizable sets of stages to include or exclude.
  // The net must have ALL of the specified stages and NONE of the specified
  // "not_stage"s to meet the rule.
  // (Use multiple NetStateRules to specify conjunctions of stages.)
  std::vector<std::string> stage_;
  std::vector<std::string> not_stage_;
};

class ParamSpec {

  public:

  // Whether to require shared weights to have the same shape, or just the same
  // count -- defaults to STRICT if unspecified.
  enum DimCheckMode {
    // STRICT (default) requires that num, channels, height, width each match.
    STRICT = 0,
    // PERMISSIVE requires only the count (num*channels*height*width) to match.
    PERMISSIVE = 1
  };

  ParamSpec():has_lr_mult_(false), decay_mult_(false) {}
  explicit ParamSpec(std::string name, enum DimCheckMode share_mode,
      float lr_mult, float decay_mult,
      bool has_lr_mult, bool has_decay_mult):
    name_(name), share_mode_(share_mode), lr_mult_(lr_mult), 
    decay_mult_(decay_mult),
    has_lr_mult_(has_lr_mult),
    has_decay_mult_(has_decay_mult)
    {}

  float lr_mult() const { return lr_mult_; }
  float decay_mult() const { return decay_mult_; }
  bool has_lr_mult() const { return has_lr_mult_; }
  bool has_decay_mult() const { return has_decay_mult_; }

  DimCheckMode share_mode() const { return share_mode_; }
  std::string name() const { return name_; }

private:
  // The names of the parameter blobs -- useful for sharing parameters among
  // layers, but never required otherwise.  To share a parameter between two
  // layers, give it a (non-empty) name.
  std::string name_;

  DimCheckMode share_mode_;

  // The multiplier on the global learning rate for this parameter.
  float lr_mult_; // = 1.0;
  bool has_lr_mult_;

  // The multiplier on the global weight decay for this parameter.
  float decay_mult_; // = 1.0;
  bool has_decay_mult_;
};

//template <typename DType>
class LayerParameter {
  public:
    inline const std::string name() const { return name_; }
    void set_name(std::string name) { name_ = name; }

    inline const std::string type() const { return type_; }
    void set_type(std::string type) { type_ = type; }

    inline int loss_weight_size() const {return loss_weight_.size();}
    inline float loss_weight(int id) const { return loss_weight_[id]; }
    inline std::vector<float> get_loss_weight_vec() const { return loss_weight_; }

    inline Phase phase() const {return phase_;}

    inline int blobs_size() const { return blob_size_; }
    inline void set_blob_size( int blob_size ) { blob_size_ = blob_size; }

    inline const std::vector<std::string>& get_bottom_vec() const { return bottom_; }
    inline std::string bottom( int id ) const { return bottom_[id]; }
    inline void set_bottom (int id, std::string name) { bottom_[id] = name; }
    inline int bottom_size() const { return bottom_.size(); }
    void add_bottom(std::string bottom_name) { bottom_.push_back(bottom_name); }

    inline const std::vector<std::string>& get_top_vec() const { return top_; }
    inline std::string top( int id ) const { return top_[id]; }
    inline int top_size() const { return top_.size(); }
    void add_top(std::string top_name) { top_.push_back(top_name); }

    inline const std::vector<bool> get_propagate_down_vec() const 
    { return propagate_down_; } 
    inline int propagate_down_size() const { return propagate_down_.size(); }

    inline const std::vector<ParamSpec> get_param_vec() const {
      return param_;
    }
    inline int param_size() const { return param_.size(); }

    NetStateRule include( int id ) const { return include_[id]; }
    inline const std::vector<NetStateRule> get_include_vec() const {
      return include_;
    }
    inline int include_size() const { return include_.size(); }

    inline const std::vector<NetStateRule> get_exclude_vec() const {
      return exclude_;
    }
    inline int exclude_size() const { return exclude_.size(); }

    NetStateRule exclude( int id ) const { return exclude_[id]; }


    //TODO
    inline const ParamSpec& param( int id ) const { return param_[id]; }
    void clear_loss_weight() { loss_weight_.clear(); }
    void add_loss_weight(float loss_weight) { loss_weight_.push_back(loss_weight); }
    void Clear() {
      bottom_.clear();
      top_.clear();
      loss_weight_.clear();
      param_.clear();
      propagate_down_.clear();
      include_.clear();
      exclude_.clear();
      blob_size_ = 0;
    }
    //TODO
    bool has_phase() const { return has_phase_; }
    void set_phase( const Phase& newphase ) { phase_ = newphase; }
    bool propagate_down( int id ) const { return propagate_down_[id]; }


#define COPY_VEC(name)\
    int name##_size = other.get_##name##_vec().size(); \
    name##_.clear(); \
    for( int i = 0; i < name##_size; ++i ){ \
      name##_.push_back(other.get_##name##_vec()[i]); \
    }

    void CopyFrom(const LayerParameter& other) {
      set_name(other.name());
      set_type(other.type());
      set_phase(other.phase());
      has_phase_ = true;
      blob_size_ = other.blobs_size();

      COPY_VEC(bottom);
      COPY_VEC(top);
      COPY_VEC(loss_weight);
      COPY_VEC(param);
      COPY_VEC(propagate_down);
      COPY_VEC(include);
      COPY_VEC(exclude);
      if(has_input_parameter_)
        input_param_.CopyFrom(other.input_param());
      if(has_inner_product_param_)
        inner_product_param_.CopyFrom(other.inner_product_param());
    }

    //TODO
    explicit LayerParameter(std::string name,
        std::string type,
        std::vector<std::string> bottom,
        std::vector<std::string> top,
        int blob_size):
      name_(name), type_(type), bottom_(bottom), top_(top),
      blob_size_(blob_size)
    {
      phase_ = TRAIN;
      has_phase_ = false;
      propagate_down_.resize(0);
      include_.resize(0);
      exclude_.resize(0);
      has_input_parameter_ = false;
      has_inner_product_param_ = false;
    }
    LayerParameter() {}
    ~LayerParameter() {}

    //for specified layers
    //Input
    void setup_input_param(const InputParameter& other) {
      input_param_.CopyFrom(other);
      has_input_parameter_ = true;
    }
    const InputParameter& input_param() const { 
      return input_param_; 
    }
    inline bool has_input_parameter() const { return has_input_parameter_; }

    //inner_product
    void setup_inner_product_param(const InnerProductParameter& other) {
      inner_product_param_.CopyFrom(other);
      has_inner_product_param_ = true;
    }
    inline const InnerProductParameter& inner_product_param() const { 
      return inner_product_param_;
    }
    inline bool has_inner_product_param() const { return has_inner_product_param_; }
    

  private:
    std::string name_;
    std::string type_;
    Phase phase_;
    bool has_phase_;

    vector<std::string> bottom_;
    vector<std::string> top_;

    std::vector<float> loss_weight_;
    std::vector<ParamSpec> param_;
    //real data should be stored in class layer
    //std::vector<shared_ptr<Blob<float> > > blobs_;
    int blob_size_;
    std::vector<bool> propagate_down_;
    std::vector<NetStateRule> include_;
    std::vector<NetStateRule> exclude_;

  //for specified layers
    InputParameter input_param_;
    bool has_input_parameter_;
    InnerProductParameter inner_product_param_;
    bool has_inner_product_param_;
};

}//end caffe
#endif
