import socket

# Define the host and port to connect to
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432  # The port used by the server

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Connect to the server
    s.connect((HOST, PORT))

    # # Send data to the server
    # message = "Hello, Server!"
    # s.sendall(message.encode('utf-8'))
    #
    # # Receive the response from the server
    # data = s.recv(1024)

    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("Exiting chat...")
            break
        s.sendall(user_message.encode('utf-8'))
        # Receive the response from the server
        data = s.recv(1024)
        print(f"Received: {data.decode('utf-8')}")