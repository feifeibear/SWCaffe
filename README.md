## Brief Introduction :
<img src="https://github.com/feifeibear/SWCaffe/blob/master/swdnnlogo.png" width = "300" height = "200" alt="swdnnlogo" align=center />
This is a caffe customized for Deep Nerual Network Training on Enbedded Devices.

### Dependencies:
1. blas
http://www.netlib.org/blas/

### Features
1. No protocbuf
2. boost with only hpp headers
3. No database for data storage, read from binary file
4. Support swBLAS and swDNN
https://github.com/THUHPGC/swDNN.git
large image channels, swDNN is used and small channels swBLAS is used

### Usage
1. Please install openBLAS into
../thirdparty/OpenBLAS/
#### MNIST + LeNet
2. please download mnist data into ../data/
http://yann.lecun.com/exdb/mnist/
2. Make test_solver
3. mpirun -n X ./test_solver for test mnist LeNet Training
#### MNIST + LSTM
2. please download mnist data into ../data/
http://yann.lecun.com/exdb/mnist/
3. Modify Makefile add
FLAGS+=-DSEQ_MNIST
2. Make test_lstm
3. mpirun -n X ./test_lstm for test mnist LSTM Training
#### IMAGENET + VGG-16
2. Please download imagenet data from 
https://pan.baidu.com/s/1bQdZcE
password: p23u
into ../data/imagenet_bin/
rename binary files into
    mean.bin
    test_data.bin
    test_label.bin
    test_mean.bin
    train_data.bin
    train_label.bin
    train_mean.bin
3. Prepare Caffemodel at ../data/serialized_caffemodel
  How to generate serialized_caffemodel?
  git checkout protobuf-loadmodel
  download VGG_ILSVRC_16_layers.caffemodel into ../data/VGG_ILSVRC_16_layers.caffemodel
  ./net_param_serialize
4. Make vggnet
5. mpirun -n X ./vggnet for test imagenet VGG Training


### Bugs
1. DataLayer is customized for mnist and imagenet
3. Only support double for Sunway
5. acc_trans is waiting to be ingegreted with code from Li Liandeng

### Developer
Jiaui Fang, Zheyu Zhang

### Contact
fang_jiarui@163.com

### current progress:
porting caffe on sw
	removing and fixing dependencies: glog, blas, hdf5, protobuffer, googletest, GPU and cudnn modules

	protobuffer for parameter communication, replace with our own code: BlobProto, LayerParameter, DataParameter, ConvolutionParameter, RecurrentParameter, LossParameter, NetParameter, SolverParameter,, etc. (21 Parameter.hpp in total)

	minor changes in source files and other header files, in order to work with other parts

	code reconstruction: big improvement on coding style and design patterns. 
	for example, for each layer_param class:
		setup_layer_param(): set param with a param object
		layer_param(): return param
		has_layer_param(): return has_param
		mutable_layer_param(): return param*(mutable)
	4 different design patterns for member variables { basic object, vector<dtype>, pointer, vector<class> }
	explicitly realize copy constructor and operator= for every class

	unit test: 26 test files in total, 25 of them have passed. test_lstm_layer is under debug.

	whole CNN test: Lenet on mnist test ok. 99% acc.

	port on single core group: done. still optimizing.

next step:
	optimize single core.
	parameter server.
	rnn support requirement from sogou. so currently helping (not in the plan)
	bigger dataset and CNN according to plan

### Construction Log 
LayerParameter.hpp

for each layer_param class:
	setup_layer_param(): set param with a param object
	layer_param(): return param
	has_layer_param(): return has_param
	mutable_layer_param(): return param*(mutable)

	NOTE: set_allocated_layer_param() is considered not necessary for now

	xxxlayer_param changed to pointer (unable to use boost::shared_ptr)

BIG BUG: LayerParameter (actually every parameter class) needs a const LayerParameter& constructor !

Changes: 4 different design patterns for member variables { basic object, vector<dtype>, pointer, vector<class> }
Please refer to LayerParameter class and follow these patterns strictly !


ConvolutionParameter.hpp

for each vector variable v:
	add_v(): add element
	v()
	v(i)
	v_size()

	filler_params are still objects (they are required fields so not using pointers)


Dangerous:
LayerParameter blobs / Blob::ToProto()
