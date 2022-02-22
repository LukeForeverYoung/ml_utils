
from utils.dict_tree import Tree


def tensor_like(ref_tensor,shape,create_fn):
    '''
    产生与ref_tensor同(device, type) 但shape自定的tensor
    '''
    return create_fn(shape,dtype=ref_tensor.dtype,device=ref_tensor.device)

def cartesian_cat(a,b):
    '''
    笛卡尔cat操作
    a: B len_a dim_a
    b: B len_b dim_b
    ->output: B len_a len_b (dim_a+dim_b)
    '''
    from einops import repeat
    import torch
    
    tmp_a=repeat(a,'B len_a dim_a -> B len_a len_b dim_a',len_b=b.shape[1])
    tmp_b=repeat(b,'B len_b dim_b -> B len_a len_b dim_b',len_a=a.shape[1])
    res=torch.cat([tmp_a,tmp_b],dim=-1)
    return res


def tree_weights(weights):
    from utils.dict_tree import Tree,dfs
    if isinstance(weights,dict):
        names=list(weights.keys())
    elif isinstance(weights,list):
        names=weights
    else:
        return
    tree=Tree()
    for name in names:
        tree.insert(name)
    dfs(tree)

if __name__ == '__main__':
    pass