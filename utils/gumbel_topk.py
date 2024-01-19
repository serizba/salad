import torch


def gumbel_topk(
    logits: torch.Tensor, 
    k: int=1, 
    tau: float=1.0, 
    eps: float=1.0E-10, 
    dim: int=-1,
):
    '''
    Discrete, differentiable, randomized implementation 
    of `top-k` function. 
    '''
    result = 0
    for i in range(k):
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + eps) + eps)
        gumbel_logits = (logits + gumbel_noise) / tau
        
        y_soft = gumbel_logits.softmax(dim=dim)
        if i == 0:
            mask = torch.zeros_like(y_soft)
        else: # no duplicate
            mask = result.clone()
            mask[mask == 1] = -torch.inf
        
        with torch.no_grad():
            y_hard = torch.zeros_like(y_soft)
            max_indices = (y_soft + mask).argmax(dim=dim, keepdim=True)
            y_hard.scatter_(dim, max_indices, 1.0)
            
        y_hard = y_hard - y_soft.detach() + y_soft
        result = result + y_hard
    return result


if __name__ == '__main__':
    from time import time
    
    SEED = 304
    DEVICE = torch.device('cpu')
    # DEVICE = torch.device('cuda:6')
    
    manual_shape = True
    if manual_shape:
        manual_shape = [512, 256]
        manual_ndim = len(manual_shape)
        t = torch.randn(manual_shape)
        print(f'{manual_ndim=}')
        print(f'{manual_shape=}')
        print(f'numel={t.numel():,}')
    
        dim, k = 1, 128
    else:
        torch.manual_seed(SEED)
        if DEVICE.type == 'cuda':
            torch.cuda.set_device(DEVICE)
            torch.cuda.manual_seed(SEED)
        
        random_ndim = torch.randint(low=1, high=5, size=(1,)).item()
        random_shape = torch.randint(low=2, high=100, size=(random_ndim,)).tolist()
        t = torch.randn(random_shape)
        print(f'{random_ndim=}')
        print(f'{random_shape=}')
        print(f'numel={t.numel():,}')
    
        dim = torch.randint(low=0, high=random_ndim, size=(1,)).item()
        k = torch.randint(low=1, high=random_shape[dim], size=(1,)).item()
    print(f'{dim=}, {k=}\n')
    
    start = time()
    r = gumbel_topk(t, k=k, dim=dim)
    end = time()
    
    good = torch.all(r.sum(dim=dim) == k)
    print('Pass' if good else 'Fail')
    print(f'{end - start:.4f} seconds.')
