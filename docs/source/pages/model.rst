How to define a model
=====================

.. testcode::

    from chebai.models.base import JCIBaseNet
    import torch

    class MyModel(JCIBaseNet):
        def __init__(self, dims):
            super().__init__()
            self.lin = torch.nn.Linear(dims, dims+1)

        def forward(self, x):
            return self.lin(x)

    model = MyModel(5)
    inp = torch.rand((1,5))
    result = model(inp)
    print(result.shape)

.. testoutput::

    torch.Size([1, 6])