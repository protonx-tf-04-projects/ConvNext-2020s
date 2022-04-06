from .block import resblock

def stage(input, filter_num, num_block, use_downsample=True, use_bottleneck=False,stage_idx=-1):
  ''' -- Stacking Residual Units on the same stage

  Args:
    filter_num: the number of filters in the convolution used during stage
    num_block: number of `Residual Unit` in a stage
    use_downsample: Down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
    use_bottleneck: type of block: basic or bottleneck
    stage_idx: index of current stage
  '''
  net = resblock(input = input, filter_num = filter_num, stride = 2 if use_downsample else 1, use_bottleneck = use_bottleneck, stage_idx = stage_idx, block_idx = 1)
    
  for i in range(1, num_block):
    net = resblock(input = net, filter_num = filter_num,stride = 1,use_bottleneck = use_bottleneck,stage_idx = stage_idx, block_idx = i+1)

  return net