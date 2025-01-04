import socket

# Chessboard constants
START_X = 0
START_Y = 0
SQUARE_SIZE = 50
GRID_SIZE = 8

# Robot server address
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 5000

def send_coordinates(s, x, y):
    """Send x and y coordinates to the robot over a socket connection."""
    try:
        # Send x-coordinate
        s.sendall(str(x).encode())
        ack = s.recv(1024).decode()
        print(f"Robot acknowledgment: {ack}")

        # Send y-coordinate
        s.sendall(str(y).encode())
        ack = s.recv(1024).decode()
        print(f"Robot acknowledgment: {ack}")
    except Exception as e:
        print(f"Error communicating with the robot: {e}")

if __name__ == "__main__":
    spots = [
        [2, 7],
        [5, 2],
        [3, 1],
        [6, 4],
        [5, 5]
    ]

    # Open socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((ROBOT_IP, ROBOT_PORT))
            print(f"Connected to robot at {ROBOT_IP}:{ROBOT_PORT}")

            # Send multiple coordinates
            for spot in spots:
                x, y = spot
                x = START_X + x * SQUARE_SIZE
                y = START_Y + y * SQUARE_SIZE
                print(f"Sending coordinates: ({x}, {y})")
                send_coordinates(s, x, y)
        except Exception as e:
            print(f"Error communicating with the robot: {e}")