import zmq
import time


def server():
    # Create a ZeroMQ Context
    context = zmq.Context()

    # Create a ROUTER socket
    socket = context.socket(zmq.ROUTER)

    # Set a unique identity for the socket (here, we're using "LAPTOP" as the identity)
    socket.identity = u"LAPTOP".encode('ascii')

    # Bind the socket to a TCP address
    socket.bind("tcp://*:5555")

    # Wait for a message from the Raspberry Pi
    message = socket.recv_string()
    print("Received message: ", message)

    # Respond with "World"
    socket.send_string("World")

    # Give the socket time to send the message before closing
    time.sleep(1)
    context.term()
    socket.close()


if __name__ == "__main__":
    server()
