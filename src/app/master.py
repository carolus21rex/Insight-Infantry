import zmq
import time


def initialize_zmq_server():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.identity = u"LAPTOP".encode('ascii')
    socket.bind("tcp://*:5555")

    return context, socket


def receive_message(socket):
    while True:  # Listen for messages indefinitely
        message = socket.recv_string()
        print("Received message: ", message)
        time.sleep(1)  # Give a small pause for ensuring completed reception before next task


def close_zmq_server(context, socket):
    context.term()
    socket.close()


def server():
    context, socket = initialize_zmq_server()

    try:
        while True:  # Runs until interrupted
            received_data = receive_message(socket)

            # Could spawn a new thread or process to run these asynchronously,
            # if your AI library and GUI update allows for it and if they are thread-safe.
            processed_data = AI_process(received_data)  # Function to call the AI component and process the data
            GUI_update(processed_data)  # Function to update the GUI with the new data

    except KeyboardInterrupt:  # Capture a keyboard interrupt, to close the server gracefully
        print("\nStopping server.")
    finally:
        close_zmq_server(context, socket)  # Always close the connections on terminating the server
