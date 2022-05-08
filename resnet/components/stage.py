from .block import resblock, downsampleblock

def downsample(input, filter_num, block_idx, stage_idx=-1):
    ''' -- Stacking Residual Units on the same stage

    Args:
      filter_num: the number of filters in the convolution used during stage
      num_block: number of `Residual Unit` in a stage
      stage_idx: index of current stage
    '''

    net = input
    net = downsampleblock(input=net, filter_num=filter_num,
                       stage_idx=stage_idx, block_idx=block_idx)

    return net

def stage(input, filter_num, num_block, stage_idx=-1):
    ''' -- Stacking Residual Units on the same stage

    Args:
      filter_num: the number of filters in the convolution used during stage
      num_block: number of `Residual Unit` in a stage
      stage_idx: index of current stage
    '''

    net = input
    for i in range(num_block):
        net = resblock(input=net, filter_num=filter_num,
                       stage_idx=stage_idx, block_idx=i+1)

    return net
