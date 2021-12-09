from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch.utils.dlpack
import kernel as gpk

from gnn_lib import GNNLIB

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)

class MySpMM(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)
        csr_sp_mat = sp_mat.to_sparse_csr()

        # return torch.mm(sp_mat, dense_mat)
        return g_spmm(csr_sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            sp_mat_dense = sp_mat.to_dense()
            sp_mat_dense = sp_mat_dense.t()
            sp_mat_t = sp_mat_dense.to_sparse_csr()
            grad_matrix2 = g_spmm(sp_mat_t, grad_output)
            grad_matrix2 = Variable(grad_matrix2)
            # grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def g_spmm(x1, x2):
    assert (x1.layout == torch.sparse_csr)
    dim0 = x1.size()[0]
    dim1 = x2.size()[1]

    # --- get input data --- #
    csr_neber_index = x1.crow_indices()
    # print("neber index:", csr_neber_index.dtype)
    csr_neber = x1.col_indices()
    # print("neber:", csr_neber.data)
    csr_neber_value = x1.values()
    # print("neber value:", csr_neber_value.data)
    w_value = x2.data
    # print("dense value:", w_value.data)

    # declare the output tensor here
    if torch.cuda.is_available() and x1.data.device != 'cpu':
        res = torch.zeros(dim0, dim1)
        res = res.cuda()

    # print("output value:",res.data)
    # input()
    csr_neber_index_dl = torch.utils.dlpack.to_dlpack(csr_neber_index)
    csr_neber_dl = torch.utils.dlpack.to_dlpack(csr_neber)
    csr_neber_value_dl = torch.utils.dlpack.to_dlpack(csr_neber_value)
    w_value_dl = torch.utils.dlpack.to_dlpack(w_value)
    res_dl = torch.utils.dlpack.to_dlpack(res)

    gpk.gspmm(csr_neber_index_dl, csr_neber_dl, csr_neber_value_dl, w_value_dl, res_dl)

    return res

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)
