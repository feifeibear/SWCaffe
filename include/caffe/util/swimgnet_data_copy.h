#ifndef SWIMGNET_DATA_COPY_H_
void swimgnet_data_copy_f(unsigned char* _data, char* _labels, float* _top0, float* _top1, float* _mean, int _batch_size, int n_cols, int n_rows);
void swimgnet_data_copy_d(unsigned char* _data, char* _labels, double* _top0, double* _top1, double* _mean, int _batch_size, int n_cols, int n_rows);
#endif
