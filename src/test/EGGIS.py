import unittest
import torch
from src.model.EGGIS_model import Net  # Import your Net class from your_module.py

class TestNet(unittest.TestCase):
    def setUp(self):
        self.net = Net(num_classes=3)  # Initialize your Net class with num_classes=3

    def test_forward(self):
        # Create a random input tensor of size (batch_size, channels, height, width)
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 128, 128)

        # Pass the input tensor through the network
        output = self.net(input_tensor)

        # Check if the output tensor has the correct shape
        self.assertEqual(output.shape, torch.Size([batch_size, 3]))
        print("Sum along dimension 1:", output.sum(dim=1))

        # Check if the output tensor sums to 1 along dimension 1 (indicating probabilities)
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.tensor([1.0, 1.0])))

if __name__ == '__main__':
    unittest.main()
