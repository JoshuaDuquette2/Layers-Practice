import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  outHeight = 1 + (H + 2 * pad - HH) // stride
  outWidth  = 1 + (W + 2 * pad - WW) // stride
  output    = np.zeros((N, F, outHeight, outWidth))

  padding = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), "constant")
  h_pad, w_pad = padding.shape[2], padding.shape[3]

  w_row = w.reshape(F, C*HH*WW)
  x_col = np.zeros((C*HH*WW, outHeight*outWidth))
  for i in range(0, N):
    pos = 0
    for j in range(0, h_pad-HH+1, stride):
      for k in range(0, w_pad-WW+1, stride):
        x_col[:, pos] = padding[i,:,j:j+HH,k:k+WW].reshape(C*HH*WW) # Fuck off
        pos += 1
    output[i] = (w_row.dot(x_col) + b.reshape(F,1)).reshape(F, outHeight, outWidth)

  out = output.copy()

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  
  N, C, h_pad, w_pad = xpad.shape
  dx = np.zeros((N, C, h_pad - 2 * pad, w_pad - 2 * pad))
  dw, db = np.zeros(w.shape), np.zeros(b.shape)
  HH, WW = f_height, f_width

  w_row = w.reshape(F, C*HH*WW)
  x_col = np.zeros((C*HH*WW, out_height*out_width))

  for i in range(0, N):
    out_col = dout[i].reshape(F, out_height*out_width)
    w_out = w_row.T.dot(out_col)
    current_dx = np.zeros((C, h_pad, w_pad))
    pos = 0
    for j in range(0, h_pad-HH+1, stride):
      for k in range(0, w_pad-WW+1, stride):
        current_dx[:, j:j+HH, k:k+WW] += w_out[:, pos].reshape(C, HH, WW)
        x_col[:, pos] = xpad[i, :, j:j+HH, k:k+WW].reshape(C*HH*WW)
        pos += 1
    dx[i] = current_dx[:, pad:-pad, pad:-pad]
    dw   += out_col.dot(x_col.T).reshape(F,C,HH,WW)
    db   += out_col.sum(axis=1)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None

  
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  stride = pool_param['stride']
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  N, C, H, W = x.shape
  outHeight = 1 + (H - pool_height) // stride
  outWidth  = 1 + (W - pool_width) // stride

  output = np.zeros((N, C, outHeight, outWidth))
  for i in range(0, N):
    out_col = np.zeros((C, outHeight*outWidth))
    pos = 0
    for j in range(0, H - pool_height + 1, stride):
      for k in range(0, W - pool_width + 1, stride):
        pool = x[i, :, j:j+pool_height, k:k+pool_width].reshape(C, pool_height*pool_width)
        out_col[:, pos] = pool.max(axis=1)
        pos += 1
    output[i] = out_col.reshape(C, outHeight, outWidth)

  out = output.copy()

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  
  N, C, outHeight, outWidth = dout.shape
  H, W = x.shape[2], x.shape[3]

  dx = np.zeros(x.shape)

  for i in range(0, N):
    dout_row = dout[i].reshape(C, outHeight*outWidth)
    pos = 0
    for j in range(0, H - pool_height + 1, stride):
      for k in range(0, W - pool_width + 1, stride):
        pool = x[i, :, j:j+pool_height, k:k+pool_width].reshape(C, pool_height*pool_width)
        max_pool_idx = pool.argmax(axis=1)
        current_dout = dout_row[:, pos]
        pos += 1

        pool_dx = np.zeros(pool.shape)
        pool_dx[np.arange(C), max_pool_idx] = current_dout
        dx[i, :, j:j+pool_height, k:k+pool_width] += pool_dx.reshape(C, pool_height, pool_width)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = x.shape
  x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = dout.shape
  dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta