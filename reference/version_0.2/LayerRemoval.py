#Example automated layer removal 

from caffe.proto import caffe_pb2
from google.protobuf import text_format

def auto_remove_layers(prototxt_path, layer_types_to_remove):
    # Load the original prototxt file
    net = caffe_pb2.NetParameter()
    with open(prototxt_path) as f:
        text_format.Merge(f.read(), net)

    # Automatically identify layers to remove based on type
    layers_to_remove = [layer.name for layer in net.layer if layer.type in layer_types_to_remove]

    # Remove identified layers
    net.layer[:] = [layer for layer in net.layer if layer.name not in layers_to_remove]

    # Save the modified prototxt
    with open('modified_model_auto.prototxt', 'w') as f:
        f.write(text_format.MessageToString(net))

# Specify the path to your original .prototxt file
prototxt_path = 'path/to/your/original_model.prototxt'
# Specify layer types to remove, e.g., 'Dropout', 'ReLU'
layer_types_to_remove = ['Dropout', 'ReLU']
auto_remove_layers(prototxt_path, layer_types_to_remove)
