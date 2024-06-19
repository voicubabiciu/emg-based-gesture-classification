import socket
import threading
import time
from threading import Thread

from rx.subject import Subject

from src.domain.emg_payload import EMGPayload


class UDPClientService(Thread):
    def __init__(self, host, port):
        super().__init__()
        self.source = None
        print(f'Initializing UDP Client on {host} and port {port}')
        self.host = host
        self.port = int(port)
        # Create a UDP socket at client side
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        try:
            self.sock.bind((self.host, self.port))
        except:
            print(f'Failed to bind to {self.host}:{self.port}')
        self.data_stream = Subject()
        self.sock.settimeout(5)

        self.running = True
        print('UDP Client initialized')

        self.start()
        print('UDP Client started')

    def run(self) -> None:
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                self.data_stream.on_next(EMGPayload.from_json(data.decode('utf-8')))
            except socket.timeout:
                # Handle timeout exception here
                # print("Timeout occurred. No data received within 5 seconds")
                pass

        print('UDP Client stopped')

    def stop(self):
        print('UDP Client stop initialized')
        self.running = False
