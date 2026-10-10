import gc
import http.client
import socket
import threading
import time
import unittest
import weakref
from http.server import BaseHTTPRequestHandler

from vllm_apple.bounded_http import BoundedHTTPServer, HeaderDeadlineMixin


class Handler(HeaderDeadlineMixin, BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'ok')

    def log_message(self, *args):
        pass


class SmallServer(BoundedHTTPServer):
    maximum_connections = 2
    header_deadline_seconds = 0.3


class BoundedHTTPTests(unittest.TestCase):
    def test_finished_handlers_are_released_without_cyclic_gc(self):
        references = []

        class TrackedHandler(Handler):
            def __init__(self, *args, **kwargs):
                references.append(weakref.ref(self))
                super().__init__(*args, **kwargs)

        server = SmallServer(('127.0.0.1', 0), TrackedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(8):
                connection = http.client.HTTPConnection(*server.server_address, timeout=2)
                try:
                    connection.request('GET', '/')
                    response = connection.getresponse()
                    self.assertEqual(response.read(), b'ok')
                finally:
                    connection.close()
                deadline = time.monotonic() + 2
                while server.resource_snapshot()['active']:
                    if time.monotonic() >= deadline:
                        self.fail('handler did not finish')
                    time.sleep(.005)
            self.assertEqual(len(references), 8)
            self.assertFalse(any(reference() is not None for reference in references),
                             'deadline callback retained finished handlers')
        finally:
            if enabled:
                gc.enable()
            server.shutdown()
            server.server_close()
            thread.join(2)
            gc.collect()

    def test_trickle_headers_are_bounded_and_slots_recover(self):
        server = SmallServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        clients = []
        try:
            for _ in range(2):
                client = socket.create_connection(server.server_address, timeout=2)
                client.sendall(b'GET / HTTP/1.1\r\nHost:')
                clients.append(client)
            deadline = time.monotonic() + 2
            while server.resource_snapshot()['active'] != 2:
                if time.monotonic() > deadline:
                    self.fail('connections were not admitted')
                time.sleep(.005)
            third = socket.create_connection(server.server_address, timeout=2)
            try:
                self.assertIn(b'503', third.recv(1024))
            finally:
                third.close()
            for _ in range(3):
                time.sleep(.06)
                for client in clients:
                    try:
                        client.sendall(b'x')
                    except (BrokenPipeError, ConnectionResetError):
                        # The absolute deadline may expire while the test thread
                        # is descheduled. Expiration and recovery remain required.
                        pass
            deadline = time.monotonic() + 2
            while server.resource_snapshot()['active']:
                if time.monotonic() > deadline:
                    self.fail('header deadline did not reclaim connections')
                time.sleep(.005)
            snapshot = server.resource_snapshot()
            self.assertEqual(snapshot['peak'], 2)
            self.assertEqual(snapshot['rejected'], 1)
            self.assertEqual(snapshot['header_expirations'], 2)
            connection = http.client.HTTPConnection(*server.server_address, timeout=2)
            try:
                connection.request('GET', '/')
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b'ok')
            finally:
                connection.close()
        finally:
            for client in clients:
                client.close()
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertFalse(thread.is_alive())
