from setup_validation import setup_validate

import torch


def fn(x, y):
    a = torch.sin(x).cuda()
    b = torch.sin(y).cuda()
    return a + b

def omit(): 
    setup_validate()


if __name__ == '__main__':
    omit()