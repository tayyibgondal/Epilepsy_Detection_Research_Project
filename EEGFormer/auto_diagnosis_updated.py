import logging
import time
from copy import copy
import sys

import numpy as np
import scipy
from numpy.random import RandomState
import resampy
import torch
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn
#from torch.nn import Identity

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)

from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity

from dataset import DiagnosisSet
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def create_set(X, y, inds):
    """
    Parameters:	
    X (3darray or list of 2darrays) – The input signal per trial.
    y (1darray or list) – Labels for each trial.
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return (torch.tensor(new_X).double().to(device), torch.tensor(new_y).double().to(device))


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        '''
        X: 3d array
        y: numpy list
        '''
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        # print(all_inds)
        # print(test_inds)
        # print(valid_inds)
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set
    
# splitter = TrainValidTestSplitter(3, 2, True)
# splitter.split(x, y)

class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        '''
        X: 3d array
        y: numpy list
        '''
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.array_equal(np.sort(np.union1d(train_inds, valid_inds)),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set
    
# ============================================
# 1DCNN BLOCK
# ============================================
class CNN1D(nn.Module):
    def __init__(self, S=21, L=150, C=120):
        super().__init__()
        self.S = S  # no. of channels
        self.L = L  # no. of sampled points
        self.C = C  # depth of convolutional dimension
        self.conv_layer_1 = nn.Conv1d(1, C, kernel_size=4)
        self.conv_layer_2 = nn.Conv1d(C, C, kernel_size=4)
        self.conv_layer_3 = nn.Conv1d(C, C, kernel_size=4)
        self.conv_layer_4 = nn.Conv1d(C, C, kernel_size=4)
        self.conv_layer_1.double()  # Update the data type of the convolutional layer weights to torch.float64
        self.conv_layer_2.double()
        self.conv_layer_3.double()
        self.conv_layer_4.double()

    def forward(self, x):
        outputs = []
        for i in range(self.S):
            input_row = x[:, i:i+1, :]  # (batch_size=1, channels=1, length=num_columns)
            output_tensor = self.conv_layer_1(input_row)
            output_tensor = self.conv_layer_2(output_tensor)
            output_tensor = self.conv_layer_3(output_tensor)
            output_tensor = self.conv_layer_4(output_tensor)
            outputs.append(output_tensor.unsqueeze(1))

        output_tensor = torch.cat(outputs, dim=1)
        return output_tensor
    
# ============================================
# REGIONAL ENCODER
# ============================================
class RegionalHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, C, L):
        # Parameters are head_size(d), no. of tokens(C), and input embedding size(L)
        super().__init__()
        self.block_size = C
        self.n_embed = L
        self.head_size = head_size 
        self.key = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S_, C_, L_ = x.shape
        x = x.view(S_, B, C_, L_)  # (B, T, C, L)
        matrices = []

        for spatial_mat in x:
            inp = spatial_mat
            # Below this, T is not the original T, but the head size
            k = self.key(inp)   # (B, C, T)
            q = self.query(inp) # (B, C, T)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, C, T) @ (B, T, C) -> (B, C, C)
            wei = F.softmax(wei, dim=-1) # (B, C, C)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(inp) # (B, C, T)
            out = wei @ v # (B, C, C) @ (B, C, T) -> (B, C, T)
            matrices.append(out.tolist())

        matrices = torch.tensor(matrices)
        out = matrices.view(B, S_, C_, self.head_size)
        return out.double()
    
class RegionalMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, C, L):
        super().__init__()
        self.heads = nn.ModuleList([RegionalHead(head_size, C, L) for _ in range(num_heads)])
        self.proj = nn.Linear(L, L).double()
        self.dropout = nn.Dropout(0.001)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1).to(device)
        # out = self.dropout(self.proj(out)) # Instead of this line, we proceed as below:

        # Implementing projection layer after the multihead attention module
        b, s, c, l = out.shape
        out = out.view(s, b, c, l)

        matrices = []
        for inp in out:
            matrix = self.dropout(self.proj(inp))
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices)
        matrices = matrices.view(b, s, c, l).to('cuda')

        return matrices.double()
    
class FeedFowardRegional(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, L): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, 4*L), 
            nn.ReLU(),
            nn.Linear(4*L, L),
            nn.Dropout(0.001),
        ).double()

    def forward(self, x):
        b, s, c, l = x.shape
        x = x.view(s, b, c, l)

        matrices = []
        for inp in x:
            matrix = self.net(inp)
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices).to(device)
        # s_, b_, c_, l_ = matrices.shape
        matrices = matrices.view(b, s, c, l)


        return matrices.double()
    
class BlockRegional(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, L, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        D = L // n_head
        self.sa = RegionalMultiHeadAttention(n_head, D, C, L)
        self.ffwd = FeedFowardRegional(L)
        self.ln1 = nn.LayerNorm(L).double()
        self.ln2 = nn.LayerNorm(L).double()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# ============================================
# SHAPE CHANGER FOR COMPATIBILITY WITH SYNCHRONOUS ENCODER
# ============================================
def RegionalToSynchronousShapeShifter(tensor):
    b, s, c, l = tensor.shape
    return tensor.view(b, c, s, l)

# ============================================
# SYNCHRONOUS ENCODER COMPONENTS
# ============================================
class SynchronousHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, S, L):
        # Parameters are head_size(d), no. of tokens(C), and input embedding size(L)
        super().__init__()
        self.block_size = S
        self.n_embed = L
        self.head_size = head_size 
        self.key = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        b, c, s, l= x.shape
        x = x.view(c, b, s, l)  # (C, B, S, L)
        matrices = []

        for spatial_mat in x:
            inp = spatial_mat
            k = self.key(inp)   # (B, S, D)
            q = self.query(inp) # (B, S, D)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, S, D) @ (B, D, S) -> (B, S, S)
            wei = F.softmax(wei, dim=-1) # (B, S, S)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(inp) # (B, S, D)
            out = wei @ v # (B, S, S) @ (B, S, D) -> (B, S, D)
            matrices.append(out.tolist())

        matrices = torch.tensor(matrices)
        out = matrices.view(b, c, s, self.head_size)
        return out.double().to(device)
    
class SynchronousMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, S, L):
        super().__init__()
        self.heads = nn.ModuleList([SynchronousHead(head_size, S, L) for _ in range(num_heads)])
        self.proj = nn.Linear(L, L).double()
        self.dropout = nn.Dropout(0.001)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1).to('cuda')
        # out = self.dropout(self.proj(out)) # Instead of this line, we proceed as below:

        # Implementing projection layer after the multihead attention module
        b, c, s, l = out.shape
        out = out.view(c, b, s, l)

        matrices = []
        for inp in out:
            matrix = self.dropout(self.proj(inp))  # inp is (B, S, L)
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices)
        matrices = matrices.view(b, c, s, l)

        return matrices.double().to(device)
    
class FeedFowardSync(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, L): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, 4*L), 
            nn.ReLU(),
            nn.Linear(4*L, L),
            nn.Dropout(dropout),
        ).double()

    def forward(self, x):
        b, c, s, l = x.shape
        x = x.view(c, b, s, l)

        matrices = []
        for inp in x:
            matrix = self.net(inp)
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices)
        # s_, b_, c_, l_ = matrices.shape
        matrices = matrices.view(b, c, s, l)


        return matrices.double().to(device)
    
class BlockSync(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, L, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        D = L // n_head
        self.sa = SynchronousMultiHeadAttention(n_head, D, S, L)
        self.ffwd = FeedFowardSync(L)
        self.ln1 = nn.LayerNorm(L).double()
        self.ln2 = nn.LayerNorm(L).double()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# ============================================
# TEMPORAL ENCODER COMPONENTS
# ============================================
class TemporalTransformer(nn.Module):
    def __init__(self, S, C, L, M):
        super(TemporalTransformer, self).__init__()
        self.C = C  # Number of channels
        self.L = L  # Original temporal dimensionality
        self.S = S  # Spatial dimension
        self.M = M  # Compressed dimensionality
        
        self.patch_size = self.C * self.S  # Patch size
        self.M_linear = nn.Linear(self.patch_size, self.patch_size).double()  # Learnable matrix M
        
    def forward(self, z5):
        # z5: (B, C, S, D) input tensor
        # Recuce the temporal dimension to M
        z5_averaged = self.reduce_temporal_dimension(z5, self.M) # (B, C, S, M)
        # Reshape the tensor to B, M, S*C
        z5_reshaped = z5_averaged.reshape(z5.shape[0], -1, self.S*self.C)  # B, M, S*C
        # Get latent vectors out of the current tensor
        latent = self.M_linear(z5_reshaped) # (B, M, S*C)
        return latent
    
    def reduce_temporal_dimension(self, input_tensor, M):
        # input_tensor: (B, C, S, L) input tensor
        # M: Compressed dimensionality

        # Reshape the tensor to 3D
        reshaped_tensor = input_tensor.view(-1, input_tensor.size(2), input_tensor.size(3))  # Shape: (B*C, S, L)

        # Calculate the mean along the last dimension (L)
        averaged_tensor = torch.mean(reshaped_tensor, dim=-1)  # Shape: (B*C, S)

        # Resize the tensor to have the desired compressed dimensionality (M)
        resized_tensor = torch.nn.functional.interpolate(averaged_tensor.unsqueeze(-1), size=M, mode='linear', align_corners=False)
        resized_tensor = resized_tensor.squeeze(-1)

        # Reshape back to 4D
        output_tensor = resized_tensor.view(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), M)  # Shape: (B, C, S, M)

        return output_tensor

class HeadTemporal(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embed):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False).double()
        self.query = nn.Linear(n_embed, head_size, bias=False).double()
        self.value = nn.Linear(n_embed, head_size, bias=False).double()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttentionTemporal(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embed):
        super().__init__()
        self.n_embed = n_embed
        self.heads = nn.ModuleList([HeadTemporal(head_size, self.n_embed) for _ in range(num_heads)])
        self.proj = nn.Linear(self.n_embed, self.n_embed).double()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFowardTemporal(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        ).double()

    def forward(self, x):
        return self.net(x)

class TemporalBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttentionTemporal(n_head, head_size, n_embd)
        self.ffwd = FeedFowardTemporal(n_embd)
        self.ln1 = nn.LayerNorm(n_embd).double()
        self.ln2 = nn.LayerNorm(n_embd).double()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# UTILITY FUNCTION NEEDED FOR TEMPORAL ENCODER
def product_of_2_least_common_factors(num):
    factors = []
    
    # Find all factors of the number
    for i in range(1, num + 1):
        if num % i == 0:
            factors.append(i)
        if len(factors) == 3:
            break
    
    ans = 1
    for factor in factors:
        ans = ans * factor
    
    return ans

# ============================================
# DECODER
# ============================================
class Decoder(nn.Module):
    def __init__(self, B, M, S, C):
        super(Decoder, self).__init__()
        self.B = B
        self.M = M
        self.S = S
        self.C = C

        # Define the layers
        # Define the 1D convolutional filter - captures info along the convolutional dimension
        self.l1_filter = nn.Conv1d(M*S, M*S, kernel_size=C).double()
        # Define the l2 filter - captures info along spatial dimension
        self.l2_filter = nn.Conv1d(M, M, kernel_size=S).double()
        # PREDICTION NEURAL NETWORK
        self.layer0 = nn.Linear(M, 256).double()
        self.layer1 = nn.Linear(256, 64).double()
        self.layer2 = nn.Linear(64, 1).double()
        self.leaky_relu = nn.LeakyReLU().double()
        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x):
        x = self.encoder_to_decoder_shape_transition(x)

        # Reshape from (B, M, S, C) to (B, M*S, C)
        x = x.view(self.B, self.M*self.S, self.C)
        # Apply the convolutional filter
        x = self.l1_filter(x)  # reduces C dimension to 1
        # Reshape the output tensor back to the desired shape (B, M, S)
        x = x.view(self.B, self.M, self.S)
        # apply
        x = self.l2_filter(x)  # this filter reduces s dimension to 1
        # Reshape
        b, m, s =  x.shape 
        x = x.view(b, m*s)  # (B, M)
    
        # Pass the input through the layers with activations
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        x = self.leaky_relu(x)
        x = self.sigmoid(x)
        return x

    def encoder_to_decoder_shape_transition(self, matrix):
        '''this function reshapes the oupput of encoder so that it is
        suitable for the decoder'''
        matrix = matrix.view(B, M, S, C)
        return matrix
    
# ============================================
# EEGFORMER
# ============================================
class EEGFormer(nn.Module):
    def __init__(self, B, S, C, L, M):
        super().__init__()
        self.B = B
        self.S = S
        self.C = C
        self.L = L
        self.M = M
        self.position_embedding_table = nn.Embedding(L, 1)
        self.conv1d_layer = CNN1D(S=S, L=L, C=C)
        self.br = BlockRegional(L, num_heads)
        self.bs = BlockSync(L, num_heads)
        self.temporal = TemporalTransformer(S, C, L, M=M) 
        self.bt = TemporalBlock(S*C, n_head=product_of_2_least_common_factors(S*C))  # nembd, nhead
        self.decoder = Decoder(B, M, S, C)
        self.position_embedding_table = nn.Embedding(L, 1)

    def forward(self, x, targets=None):
        # x is eeg segment
        # x = x + self.position_embedding_table(x.long()).squeeze().double()
        x = self.conv1d_layer(x)
        x = pad_tensor(x, dim=3, length=num_cols//128)
        x = self.br(x)
        x = RegionalToSynchronousShapeShifter(x)
        x = self.bs(x)
        x = self.temporal(x)
        x = self.bt(x)
        x = self.decoder(x)

        if targets == None:
            loss = None
        else:
            B, cols = x.shape
            probabilities = x.view(B*cols,)
            loss = F.binary_cross_entropy(probabilities, targets)   
                     
        return x, loss
    
# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications
data_folders = [
    '/home/arooba/ssd/hira/nmt dataset/nmt_scalp_eeg_dataset/normal',
    '/home/arooba/ssd/hira/nmt dataset/nmt_scalp_eeg_dataset/abnormal']
n_recordings = 16  # set to an integer, if you want to restrict the set size
sensor_types = ["EEG"]
n_chans = 21
max_recording_mins = None # exclude larger recordings from training set
sec_to_cut = 60  # cut away at start of each recording
duration_recording_mins = 10  # how many minutes to use per recording
test_recording_mins = 10
max_abs_val = 800  # for clipping
sampling_freq = 100
divisor = 10  # divide signal by this
test_on_eval = True  # test on evaluation set or on training set
# in case of test on eval, n_folds and i_testfold determine
# validation fold in training set for training until first stop
n_folds = 10
i_test_fold = 9
shuffle = True
model_name = 'transformer'#shallow/deep for DNN (deep terminal local 1)
n_start_chans = 25
n_chan_factor = 2  # relevant for deep model only
input_time_length = 200 #=========
final_conv_length = 1
model_constraint = 'defaultnorm'
init_lr = 1e-3
batch_size = 64
max_epochs = 35 # until first stop, the continue train on train+valid
cuda = False
num_cols = 60032  #===========

B = 128  # batch size 
S = n_chans  # channels
C = 120  # convolutional dimension depth
L = num_cols//B  # segment length  #========== this turns out to be 469
M = L // 5  # reduced temporal dimension
num_heads = 7 # divides the temporal dim L perfectly
path_to_label_ref_csv = r"C:\Users\DELL\Downloads\tukl\Implementations\eegformer\dataset_s\new.csv"
root_dir = r"C:\Users\DELL\Downloads\tukl\Implementations\eegformer\dataset_s\*\*\*.edf"
dropout = 0.01
eval_iters = 1
eval_interval = 3
log_file_path = "loss_log.txt"
max_iters = 469  # 469 * batch_size = 60032  # for one example
learning_rate = 0.01
dropout = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_tensor(tensor, dim, length):
    tensor_shape = list(tensor.shape)
    current_length = tensor_shape[dim]

    if current_length >= length:
        return tensor

    padding_shape = tensor_shape.copy()
    padding_shape[dim] = length - current_length

    padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=device)
    padded_tensor = torch.cat((tensor, padding), dim=dim)

    return padded_tensor

# Function to calculate F1 score, precision, and recall for both classes
def calculate_metrics(predictions, targets, num_classes=2):
    predictions = torch.tensor(predictions).int()
    targets = torch.tensor(targets).int()

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for p, t in zip(predictions, targets):
        confusion_matrix[t][p] += 1

    metrics_per_class = {}
    for i in range(num_classes):
        true_positives = confusion_matrix[i][i]
        false_positives = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        false_negatives = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics_per_class[f'c{i}_Precision'] = precision
        metrics_per_class[f'c{i}_Recall'] = recall
        metrics_per_class[f'c{i}_F1_Score'] = f1_score

    return metrics_per_class

# Updated Loss Estimator
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)

        all_predictions = []
        all_targets = []
        for k in range(eval_iters):
            X, Y = get_batch(t, test=True) 
            probs, loss = model(X, Y)
            losses[k] = loss.item()

            # Convert probabilities to class predictions
            predictions = (probs > 0.5).int()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(Y.cpu().numpy())

        out[split] = {
            'loss': losses.mean(),
            **calculate_metrics(all_predictions, all_targets)  # Include metrics in the output
        }
    model.train()
    return out

def run_exp(data_folders,
            n_recordings,
            sensor_types,
            n_chans,
            max_recording_mins,
            sec_to_cut, duration_recording_mins,
            test_recording_mins,
            max_abs_val,
            sampling_freq,
            divisor,
            test_on_eval,
            n_folds, i_test_fold,
            shuffle,
            model_name,
            n_start_chans, n_chan_factor,
            input_time_length, final_conv_length,
            model_constraint,
            init_lr,
            batch_size, max_epochs,cuda, num_cols):
    # WRITE PREPROCESSING FUNCTIONS
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    preproc_functions = []
    # cut seconds from start and end
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    # crop recording to maximum duration min
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))
    # Clipping the data
    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val), fs))
    #my edit due to ValueError: Input signal length=0 is too small to resample from 250.0->100     if data.shape[1]!=0:
    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))
    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))
    preproc_functions.append(lambda arr, sampling_freq: \
            (np.pad(arr, ((0, n_chans - arr.shape[0]), (0, num_cols - arr.shape[1])),
            mode='constant', constant_values=-100), sampling_freq))

    # MAKE DATASET OBJECTS
    dataset = DiagnosisSet(n_recordings=n_recordings,
                           max_recording_mins=max_recording_mins,
                           preproc_functions=preproc_functions,
                           data_folders=data_folders,
                           train_or_eval='train',
                           sensor_types=sensor_types)
    if test_on_eval:
        if test_recording_mins is None:
            test_recording_mins = duration_recording_mins
        test_preproc_functions = copy(preproc_functions)
        test_preproc_functions[1] = lambda data, fs: (
            data[:, :int(test_recording_mins * 60 * fs)], fs)
        test_dataset = DiagnosisSet(n_recordings=n_recordings,
                                max_recording_mins=None,
                                preproc_functions=test_preproc_functions,
                                data_folders=data_folders,
                                train_or_eval='eval',
                                sensor_types=sensor_types)
        
    # LOAD DATA FROM DATASET OBJECTS
    X,y = dataset.load()
    max_shape = np.max([list(x.shape) for x in X],
                       axis=0)
    # assert max_shape[1] == int(duration_recording_mins *
                            #    sampling_freq * 60)
    if test_on_eval:
        test_X, test_y = test_dataset.load()
        max_shape = np.max([list(x.shape) for x in test_X],
                           axis=0)
        #assert max_shape[1] == int(test_recording_mins *   sampling_freq * 60)
    if not test_on_eval:
        splitter = TrainValidTestSplitter(n_folds, i_test_fold,
                                          shuffle=shuffle)
        train_set, valid_set, test_set = splitter.split(X, y)
    else:
        splitter = TrainValidSplitter(n_folds, i_valid_fold=i_test_fold,
                                          shuffle=shuffle)
        train_set, valid_set = splitter.split(X, y)
        test_set = SignalAndTarget(test_X, test_y)
        del test_X, test_y
    del X,y # shouldn't be necessary, but just to make sure

    # AT THIS POINT WE HAVE TRAIN SET, TEST SET, VALIDATION SET

    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    if model_name == 'linear':
        model = nn.Sequential()
        model.add_module("conv_classifier",
                         nn.Conv2d(n_chans, n_classes, (600,1)))
        model.add_module('softmax', nn.LogSoftmax())
        model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
    elif model_name == 'transformer':
        model = EEGFormer(B, S, C, L, M)
    else:
        assert False, "unknown model name {:s}".format(model_name)

    return train_set, valid_set, test_set, model.to(device)

t, v, te, model = run_exp(data_folders,
            n_recordings,
            sensor_types,
            n_chans,
            max_recording_mins,
            sec_to_cut, duration_recording_mins,
            test_recording_mins,
            max_abs_val,
            sampling_freq,
            divisor,
            test_on_eval,
            n_folds, i_test_fold,
            shuffle,
            model_name,
            n_start_chans, n_chan_factor,
            input_time_length, final_conv_length,
            model_constraint,
            init_lr,
            batch_size, max_epochs,cuda, num_cols)

def segment_list(original_list, segment_length):
    segmented_list = []
    for element in original_list:
        # Calculate the number of segments that can be obtained from the element
        num_segments = element.size(1) // segment_length
        # Split the element into segments of specified length along the second axis
        segments = torch.split(element[:, :num_segments * segment_length], segment_length, dim=1)
        # Append the segments to the segmented list
        segmented_list.extend(segments)
    return torch.stack(segmented_list)

def get_batch(dataset, test=False):
    global start
    if not test:
        # data set is a tuple: (list of x, list of y)
        x = segment_list([dataset[0][start]], int(60032/128))
        y = dataset[1][start].repeat(1, 128).view(128,)
        
        start += 1
    if test:
        # data set is a tuple: (list of x, list of y)
        max_i = len(dataset[0])
        ind = torch.randint(0, max_i, (1,)).item()
        x = segment_list([dataset[0][ind]], int(60032/128))
        y = dataset[1][ind].repeat(1, 128).view(128,)
    return x, y

#==================================
# TRAINING LOOP
#==================================
# Create a list to store training and validation losses
train_losses = []
val_losses = []

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
start = 0
# Training loop
print('starting the training....................')
for iter in range(len(t[0])):  # len(t[0]) reresents no. of elements in training set
    print('iter: ', iter)
    # Every once in a while, evaluate the loss, F1 score, precision, and recall on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses_and_metrics = estimate_loss()
        train_loss = losses_and_metrics['train']['loss']
        val_loss = losses_and_metrics['eval']['loss']
        print(f"Step {iter}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        # Append losses to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Log the losses and metrics to a text file
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Step {iter}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}\n")
            for metric_name, metric_value in losses_and_metrics['eval'].items():
                log_file.write(f"{metric_name.capitalize()}: {metric_value:.4f}\n")

    # Sample a batch of data
    xb, yb = get_batch(t)

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()