import unittest
import os

# Get the current directory where this script is located
current_directory = os.path.dirname(__file__)

# Discover and load all test files in the current directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(current_directory, pattern="test_*.py")

if __name__ == '__main__':
    # Run the discovered test suite
    unittest.TextTestRunner().run(test_suite)
