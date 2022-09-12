# -*- coding: utf-8 -*-
import socket
import struct
import sys

PORT = 7123

MSG_QUIT = 0
MSG_BIAS = 1
MSG_AMP = 2

mla = None


def set_dc_bias(port, bias):
    sock = _connect()
    payload = struct.pack("<QQd", MSG_BIAS, int(port), float(bias))
    sock.sendall(payload)
    sock.close()


def set_amp(port, amp):
    sock = _connect()
    payload = struct.pack("<QQQ", MSG_AMP, int(port), 1 if amp else 0)
    sock.sendall(payload)
    sock.close()


def quit():
    sock = _connect()
    payload = struct.pack("<Q", MSG_QUIT)
    sock.sendall(payload)
    sock.close()


def _connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
    sock.connect(("127.0.0.1", PORT))
    return sock


def _handle_connection(sock):
    global mla

    should_quit = False
    while True:
        buf = sock.recv(8)
        if len(buf) == 0:
            break
        msg = int.from_bytes(buf, byteorder="little", signed=False)
        if msg == MSG_QUIT:
            print("--- quitting server")
            should_quit = True
        elif msg == MSG_BIAS:
            buf = sock.recv(16)
            if len(buf) == 0:
                break
            port, bias = struct.unpack("<Qd", buf)
            print(f"--- setting dc bias on port {port} to {bias} V")
            mla.lockin.set_dc_offset(port, bias)
        elif msg == MSG_AMP:
            buf = sock.recv(16)
            if len(buf) == 0:
                break
            port, amp = struct.unpack("<QQ", buf)
            print(f"--- setting amplification on port {port} to {amp}")
            mla.analog.set_output_range(port, amp)
        else:
            print(f"*** unknown message {msg}")

    return should_quit


def _receive(sock, N):
    expected = int(N)
    buf = bytearray(expected)
    view = memoryview(buf)
    received = 0
    while received < expected:
        remaining = expected - received
        received += sock.recv_into(view[received:expected], remaining)
    return buf


if __name__ == "__main__":
    if "/home/riccardo/IntermodulatorSuite" not in sys.path:
        sys.path.append("/home/riccardo/IntermodulatorSuite")
    from mlaapi import mla_api, mla_globals

    settings = mla_globals.read_config()
    mla = mla_api.MLA(settings)
    mla.connect()
    try:
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind(("127.0.0.1", PORT))
        serversocket.listen(0)

        while True:
            (clientsocket, address) = serversocket.accept()
            clientsocket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
            if _handle_connection(clientsocket):
                break
            clientsocket.close()

        serversocket.close()
    finally:
        mla.reset()
        mla.disconnect()
