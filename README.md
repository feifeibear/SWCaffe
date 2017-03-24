## Brief Introduction :
This is a caffe customized for Deep Nerual Network Training on Enbedded Devices.

### Dependencies:
1. blas
http://www.netlib.org/blas/

### Features
1. No protocbuf
2. boost with only hpp headers
3. No database for data storage, read from binary file

### Usage
1. Please install openBLAS into 
../thirdparty/OpenBLAS/
2. please download mnist data into ../data/
http://yann.lecun.com/exdb/mnist/
2. Make
3. ./test_solver for test mnist LeNet Training

### Bugs
DataLayer is customized for mnist

### Developer
Jiaui Fang
Zheyu Zhang

### Contact
fang_jiarui@163.com

