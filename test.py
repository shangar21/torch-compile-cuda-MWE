import torch
#from torch.utils.cpp_extension import load
#
#matDxD = load(name="matDxD", sources=["mul.cpp"])
import matDxD

class SampleGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return matDxD.matDxD(a, b)

    @staticmethod
    def backward(ctx, a, b):
        return a

class SampleNet(torch.nn.Module):

    def __init__(self, n):
        super(SampleNet, self).__init__()
        self.a = torch.randn(n, n)

    def forward(self, b):
        return SampleGrad.apply(self.a, b)

n = 3
test_n = SampleNet(n)
torch.compile(test_n)
b = torch.randn(n, n)

print(test_n(b))
