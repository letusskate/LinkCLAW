import torch,sys
print('torch', getattr(torch,'__version__',None), 'has_int1', hasattr(torch,'int1'))
try:
    import torchao
    print('torchao', getattr(torchao,'__version__',None))
except Exception as e:
    print('torchao import error:', e)