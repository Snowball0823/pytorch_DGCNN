#pragma once

#include "kernel.h"
void invoke_spmm(array1d_t <int64_t> & neber_index, array1d_t <int64_t> & neber, array1d_t <float> & neber_value, array2d_t <float> & w_value, array2d_t <float> & output);

graph_t* invoke_init_graph(vid_t v_count, vid_t dst_size, vid_t* offset_csr, void* nebrs_csr, vid_t* offset_csc, void* nebrs_csc);
