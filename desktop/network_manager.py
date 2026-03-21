#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         network_manager.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/

# /GrabDrop-Desktop/network_manager.py
import json
import logging
import socket
import struct
import threading
import time
from typing import Callable, Dict, Optional

import config

logger = logging.getLogger("Network")


class NetworkManager:
    def __init__(self):
        self.on_screenshot_offer: Optional[Callable[[dict], None]] = None

        self._running = False
        self._discovered_devices: Dict[str, dict] = {}
        self._devices_lock = threading.Lock()

        self._listener_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._tcp_server_socket: Optional[socket.socket] = None

    @property
    def nearby_count(self) -> int:
        with self._devices_lock:
            return len(self._discovered_devices)

    def start(self):
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._udp_listener, daemon=True
        )
        self._listener_thread.start()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_thread.start()

        logger.info("Network manager started")

    def stop(self):
        self._running = False
        if self._tcp_server_socket:
            try:
                self._tcp_server_socket.close()
            except Exception:
                pass
        logger.info("Network manager stopped")

    # --- Heartbeat ---

    def _heartbeat_loop(self):
        self._send_heartbeat()  # immediate first
        while self._running:
            time.sleep(config.HEARTBEAT_INTERVAL_S)
            if self._running:
                self._send_heartbeat()

    def _send_heartbeat(self):
        msg = json.dumps({
            "type": config.HEARTBEAT_TYPE,
            "sender_id": config.DEVICE_ID,
            "sender_name": config.DEVICE_NAME,
            "timestamp": time.time(),
        }).encode()

        # Multicast
        try:
            sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.sendto(msg, (config.MULTICAST_GROUP, config.UDP_PORT))
            sock.close()
        except Exception as e:
            logger.debug(f"Multicast heartbeat failed: {e}")

        # Broadcast
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(msg, ("255.255.255.255", config.UDP_PORT))
            sock.close()
        except Exception as e:
            logger.debug(f"Broadcast heartbeat failed: {e}")

    # --- Device Cleanup ---

    def _cleanup_loop(self):
        while self._running:
            time.sleep(5)
            now = time.time()
            with self._devices_lock:
                stale = [
                    did
                    for did, info in self._discovered_devices.items()
                    if now - info["last_seen"] > config.DEVICE_TIMEOUT_S
                ]
                for did in stale:
                    name = self._discovered_devices[did]["name"]
                    del self._discovered_devices[did]
                    logger.info(f"Device expired: {name} ({did})")

                if stale:
                    logger.info(
                        f"Nearby devices: {len(self._discovered_devices)}"
                    )

    # --- UDP Listener ---

    def _udp_listener(self):
        sock = None
        try:
            sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Platform-specific
            try:
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
                )
            except AttributeError:
                pass  # Windows

            sock.bind(("", config.UDP_PORT))

            # Join multicast group
            try:
                group = socket.inet_aton(config.MULTICAST_GROUP)
                mreq = struct.pack("4sL", group, socket.INADDR_ANY)
                sock.setsockopt(
                    socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq
                )
            except Exception as e:
                logger.warning(f"Multicast join failed: {e}")

            # Also enable broadcast receive
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            sock.settimeout(1.0)
            logger.info(f"UDP listener started on port {config.UDP_PORT}")

            while self._running:
                try:
                    data, addr = sock.recvfrom(4096)
                    msg = data.decode("utf-8")
                    self._handle_udp_message(msg, addr[0])
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger.error(f"UDP receive error: {e}")
                        time.sleep(1)

        except Exception as e:
            logger.error(f"UDP listener failed: {e}", exc_info=True)
        finally:
            if sock:
                sock.close()

    def _handle_udp_message(self, message: str, sender_ip: str):
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            sender_id = data.get("sender_id")

            if sender_id == config.DEVICE_ID:
                return  # ignore own messages

            sender_name = data.get("sender_name", "Unknown")

            if msg_type == config.HEARTBEAT_TYPE:
                self._handle_heartbeat(sender_id, sender_name, sender_ip)

            elif msg_type == config.BROADCAST_TYPE_SCREENSHOT_READY:
                self._handle_heartbeat(sender_id, sender_name, sender_ip)

                offer = {
                    "sender_id": sender_id,
                    "sender_name": sender_name,
                    "sender_address": sender_ip,
                    "tcp_port": data["tcp_port"],
                    "file_size": data.get("file_size", 0),
                    "timestamp": data.get("timestamp", time.time()),
                }
                logger.info(
                    f"📡 Screenshot offer from {sender_name} "
                    f"({sender_ip}:{offer['tcp_port']})"
                )
                if self.on_screenshot_offer:
                    self.on_screenshot_offer(offer)

        except json.JSONDecodeError:
            logger.debug(f"Invalid JSON from {sender_ip}")
        except Exception as e:
            logger.error(f"Error handling UDP message: {e}")

    def _handle_heartbeat(
        self, sender_id: str, sender_name: str, sender_ip: str
    ):
        with self._devices_lock:
            is_new = sender_id not in self._discovered_devices
            self._discovered_devices[sender_id] = {
                "name": sender_name,
                "address": sender_ip,
                "last_seen": time.time(),
            }
            if is_new:
                count = len(self._discovered_devices)
                logger.info(
                    f"🔍 New device: {sender_name} ({sender_ip}) "
                    f"— total nearby: {count}"
                )

    # --- Screenshot Broadcast ---

    def broadcast_screenshot(self, screenshot_data: bytes):
        """Start TCP server and broadcast availability."""
        tcp_port = self._start_tcp_server(screenshot_data)

        msg = json.dumps({
            "type": config.BROADCAST_TYPE_SCREENSHOT_READY,
            "sender_id": config.DEVICE_ID,
            "sender_name": config.DEVICE_NAME,
            "tcp_port": tcp_port,
            "file_size": len(screenshot_data),
            "timestamp": time.time(),
        }).encode()

        # Multicast
        try:
            sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.sendto(msg, (config.MULTICAST_GROUP, config.UDP_PORT))
            sock.close()
        except Exception as e:
            logger.debug(f"Multicast broadcast failed: {e}")

        # Broadcast
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(msg, ("255.255.255.255", config.UDP_PORT))
            sock.close()
        except Exception as e:
            logger.debug(f"Broadcast failed: {e}")

        logger.info(f"Screenshot broadcast sent (port {tcp_port})")

    def _start_tcp_server(self, data: bytes) -> int:
        if self._tcp_server_socket:
            try:
                self._tcp_server_socket.close()
            except Exception:
                pass

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", 0))
        port = server.getsockname()[1]
        server.listen(5)
        server.settimeout(config.SCREENSHOT_OFFER_TIMEOUT_S + 5)
        self._tcp_server_socket = server

        def serve():
            try:
                for _ in range(5):
                    try:
                        client, addr = server.accept()
                        threading.Thread(
                            target=self._handle_tcp_client,
                            args=(client, data, addr),
                            daemon=True,
                        ).start()
                    except socket.timeout:
                        break
            except Exception as e:
                if self._running:
                    logger.error(f"TCP server error: {e}")
            finally:
                server.close()

        threading.Thread(target=serve, daemon=True).start()
        logger.info(f"TCP server listening on port {port}")
        return port

    def _handle_tcp_client(
        self, client: socket.socket, data: bytes, addr
    ):
        try:
            request = client.recv(64).decode().strip()
            if request == "GET":
                header = struct.pack("!I", len(data))
                client.sendall(header + data)
                logger.info(
                    f"Sent {len(data)} bytes to {addr[0]}:{addr[1]}"
                )
        except Exception as e:
            logger.error(f"TCP client handler error: {e}")
        finally:
            client.close()

    # --- Download ---

    def download_screenshot(self, offer: dict) -> Optional[bytes]:
        try:
            addr = offer["sender_address"]
            port = offer["tcp_port"]

            logger.info(f"Connecting to {addr}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((addr, port))

            sock.sendall(b"GET\n")

            # Read 4-byte length header
            header = b""
            while len(header) < 4:
                chunk = sock.recv(4 - len(header))
                if not chunk:
                    raise ConnectionError("Connection closed reading header")
                header += chunk

            length = struct.unpack("!I", header)[0]
            logger.info(f"Receiving {length} bytes...")

            # Read data
            data = b""
            while len(data) < length:
                chunk = sock.recv(min(65536, length - len(data)))
                if not chunk:
                    raise ConnectionError("Connection closed reading data")
                data += chunk

            sock.close()
            logger.info(f"Download complete ({len(data)} bytes)")
            return data

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

