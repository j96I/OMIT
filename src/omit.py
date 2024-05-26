from utils.setup_validation import setup_validate

import torch


def fn(x, y):
    a = torch.sin(x).cuda()
    b = torch.sin(y).cuda()
    return a + b


def omit(): 
    new_fn = torch.compile(fn, backend="common")
    input_tensor = torch.randn(10000).to(device="cuda:0")
    a = new_fn(input_tensor, input_tensor)
    print(a)
    


if __name__ == '__main__':
    omit()