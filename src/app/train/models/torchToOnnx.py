import torch

pytorch_model = torch.load('best.pt')

dummy_input = torch.randn(1, 3, 416, 416)  # adjust the size according to your model input
torch.onnx.export(pytorch_model, dummy_input, 'best.onnx', verbose=True)
