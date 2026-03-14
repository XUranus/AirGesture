# GrabDrop Network Protocol Specification

**Version:** 1.0  
**Status:** Stable  
**Transport:** UDP (discovery) + TCP (transfer)

## Overview

GrabDrop uses a simple, lightweight protocol for device discovery and screenshot transfer over a local area network. No central server, no internet connection, and no pairing is required.

```
┌─────────────────────────────────────────────────────┐
│                   PROTOCOL LAYERS                    │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Application Layer                             │  │
│  │  JSON messages + binary PNG data               │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Transport Layer                               │  │
│  │  UDP 9877 (discovery) + TCP dynamic (transfer) │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Network Layer                                 │  │
│  │  IPv4 LAN (multicast 239.255.77.88 + broadcast)│ │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 1. Device Discovery

### 1.1 Heartbeat

Every device sends periodic heartbeat messages to announce its presence.

**Transport:** UDP  
**Destination:** `239.255.77.88:9877` (multicast) AND `255.255.255.255:9877` (broadcast)  
**Interval:** Every 3 seconds  
**Timeout:** Device considered offline after 10 seconds without heartbeat

#### Message Format

```json
{
  "type": "HEARTBEAT",
  "sender_id": "a3f8b2c1",
  "sender_name": "Pixel_8",
  "timestamp": 1710345600.123
}
```

#### Fields

| Field | Type | Description |
|---|---|---|
| `type` | string | Always `"HEARTBEAT"` |
| `sender_id` | string | 8-character unique device ID (random UUID prefix) |
| `sender_name` | string | Human-readable device name (e.g., `Build.MODEL` on Android, `platform.node()` on desktop) |
| `timestamp` | number | Unix timestamp (seconds with millisecond precision) |

#### Device Table Management

```
On receiving a HEARTBEAT from sender_id X:
  if X == self.device_id:
    → ignore (own message)
  if X not in discovered_devices:
    → add to table with last_seen = now
    → log "New device discovered"
  else:
    → update last_seen = now

Every 5 seconds:
  for each device in discovered_devices:
    if now - last_seen > 10s:
      → remove from table
      → log "Device expired"
```

### 1.2 Dual-Path Delivery

Both multicast AND broadcast are used for reliability:

```
Sender
  ├── Multicast: 239.255.77.88:9877
  │   └── Works on most routers, efficient
  │
  └── Broadcast: 255.255.255.255:9877
      └── Fallback for networks that block multicast
```

Receivers join the multicast group AND listen for broadcast on the same port. Duplicate messages are ignored via `sender_id` matching.

### 1.3 Multicast Lock (Android-specific)

Android devices must acquire a `WifiManager.MulticastLock` to receive multicast packets:

```kotlin
val lock = wifiManager.createMulticastLock("GrabDrop")
lock.setReferenceCounted(true)
lock.acquire()
```

## 2. Screenshot Broadcast

When a device captures a screenshot, it announces availability via UDP.

### 2.1 Screenshot Offer Message

**Transport:** UDP  
**Destination:** Same as heartbeat (multicast + broadcast on port 9877)

```json
{
  "type": "SCREENSHOT_READY",
  "sender_id": "a3f8b2c1",
  "sender_name": "Pixel_8",
  "tcp_port": 54321,
  "file_size": 524288,
  "timestamp": 1710345605.456
}
```

#### Fields

| Field | Type | Description |
|---|---|---|
| `type` | string | Always `"SCREENSHOT_READY"` |
| `sender_id` | string | Sender's device ID |
| `sender_name` | string | Sender's device name |
| `tcp_port` | integer | TCP port where screenshot can be downloaded |
| `file_size` | integer | Size of screenshot in bytes |
| `timestamp` | number | When screenshot was captured (Unix timestamp) |

### 2.2 Offer Lifecycle

```
Sender captures screenshot
    │
    ├── 1. Start TCP server on random port
    │      └── server.bind(("", 0))  → OS assigns port
    │
    ├── 2. Send UDP broadcast with tcp_port
    │
    ├── 3. Wait for TCP connections (up to 15s)
    │      └── Accept up to 5 clients
    │
    └── 4. Close TCP server after timeout
```

### 2.3 Offer Expiry

- **Sender:** TCP server closes 15 seconds after broadcast
- **Receiver:** Offers older than 10 seconds are discarded from the queue

## 3. Screenshot Transfer

### 3.1 TCP Handshake

```
Receiver                          Sender
────────                          ──────
                                  TCP server listening on port P

Connect to sender_ip:P ─────────►

Send: "GET\n"           ─────────►
                                  Read request line

                         ◄───────── Send: [4 bytes big-endian uint32: length]
                         ◄───────── Send: [length bytes: PNG data]

Close connection                  Close connection
```

### 3.2 Binary Wire Format

```
┌──────────────────┬────────────────────────────┐
│  Length Header    │  PNG Image Data            │
│  4 bytes          │  variable length           │
│  big-endian u32  │  raw bytes                 │
├──────────────────┼────────────────────────────┤
│  00 08 00 00     │  89 50 4E 47 0D 0A ...    │
│  (= 524,288)     │  (PNG file bytes)          │
└──────────────────┴────────────────────────────┘
```

### 3.3 Transfer Sequence Diagram

```
Device A (GRAB)                    Device B (RELEASE)
───────────────                    ─────────────────

[1] Gesture: palm→fist
[2] Screenshot captured
[3] TCP server started (port 54321)
[4] UDP broadcast: ─────────────────► [5] Offer received
    {"type":"SCREENSHOT_READY",           • Queued in offer list
     "tcp_port":54321,                    • Wait for RELEASE gesture
     "file_size":524288}
                                   [6] Gesture: fist→palm
                                   [7] Match offer from queue
                                   [8] TCP connect ◄──────────
                                   [9] Send "GET\n" ──────────►
[10] Send length (4 bytes) ◄──────
[11] Send PNG data         ◄──────
                                   [12] Receive complete
                                   [13] Save to storage
                                   [14] Show notification
                                   [15] Open gallery
```

## 4. Retroactive Matching

A RELEASE gesture may occur *before* the screenshot offer arrives (network latency). The protocol handles this with retroactive matching:

```
Time ──────────────────────────────────────────►

Case 1: Normal (Offer first, then RELEASE)
  t=0.0  Offer received → queued
  t=2.5  RELEASE gesture → match from queue ✅

Case 2: Retroactive (RELEASE first, then Offer)
  t=0.0  RELEASE gesture → no offer → record timestamp
  t=0.8  Offer received → check: RELEASE was 0.8s ago (< 3s)
         → retroactive match ✅ → auto-download

Case 3: Expired (too old)
  t=0.0  Offer received → queued
  t=12.0 RELEASE gesture → offer is 12s old (> 10s timeout)
         → expired, discarded ❌
```

**Retroactive match window:** 3 seconds

## 5. Multi-Device Behavior

### 5.1 Multiple Receivers

When Device A broadcasts a screenshot, ALL devices on the network receive the offer. Each device independently decides whether to accept (RELEASE gesture). Multiple devices can download the same screenshot simultaneously — the TCP server accepts up to 5 connections.

```
Device A (sender)
  │
  ├── UDP broadcast ──► Device B: receives offer, does RELEASE → downloads ✅
  ├── UDP broadcast ──► Device C: receives offer, does RELEASE → downloads ✅
  └── UDP broadcast ──► Device D: receives offer, ignores ────── no action
```

### 5.2 Multiple Senders

If multiple devices send screenshots before a RELEASE, the receiver uses the **most recent** offer:

```
t=0.0  Offer from Device A → queued
t=1.0  Offer from Device B → queued
t=2.0  RELEASE on Device C → takes Device B's offer (newest)
       Device A's offer discarded
```

## 6. Constants Reference

| Parameter | Value | Description |
|---|---|---|
| `UDP_PORT` | `9877` | Shared discovery port |
| `MULTICAST_GROUP` | `239.255.77.88` | Multicast address for discovery |
| `HEARTBEAT_INTERVAL` | 3s | Time between heartbeat messages |
| `DEVICE_TIMEOUT` | 10s | Remove device after N seconds without heartbeat |
| `SCREENSHOT_OFFER_TIMEOUT` | 10s | Offers expire after this duration |
| `RETROACTIVE_MATCH_WINDOW` | 3s | RELEASE before offer matching window |
| `TCP_SERVER_TIMEOUT` | 15s | TCP server closes after this duration |
| `TCP_MAX_CLIENTS` | 5 | Max simultaneous downloads per offer |
| `IMAGE_FORMAT` | PNG | Screenshot encoding format |
| `HEADER_SIZE` | 4 bytes | Big-endian uint32 length prefix |

## 7. Implementing a New Client

### Minimum Viable Implementation

To build a GrabDrop-compatible client, implement:

1. **UDP Listener** on port 9877 (multicast + broadcast)
2. **UDP Sender** for heartbeats every 3 seconds
3. **JSON Parser** for `HEARTBEAT` and `SCREENSHOT_READY` messages
4. **TCP Client** that connects, sends `"GET\n"`, reads `[4-byte-len][data]`
5. **TCP Server** on a random port, serves PNG data with same wire format

### Pseudo-Code: Minimal Receiver

```python
# 1. Listen for offers
sock = udp_socket(port=9877)
while True:
    msg = sock.recv()
    data = json.parse(msg)
    
    if data.type == "SCREENSHOT_READY":
        # 2. Download immediately (or wait for gesture)
        tcp = tcp_connect(data.sender_address, data.tcp_port)
        tcp.send("GET\n")
        length = tcp.recv_uint32_bigendian()
        png_data = tcp.recv_exact(length)
        
        # 3. Save
        file.write("received.png", png_data)
```

### Pseudo-Code: Minimal Sender

```python
# 1. Capture screenshot
png_data = screenshot()

# 2. Start TCP server
server = tcp_listen(port=0)  # random port
port = server.local_port

# 3. Broadcast offer  
udp_send(port=9877, json={
    "type": "SCREENSHOT_READY",
    "sender_id": my_id,
    "sender_name": my_name,
    "tcp_port": port,
    "file_size": len(png_data)
})

# 4. Serve data
client = server.accept()
request = client.recv_line()  # "GET"
client.send_uint32_bigendian(len(png_data))
client.send(png_data)
```

## 8. Security Considerations

⚠️ **GrabDrop v1.0 has NO encryption or authentication.**

- All data travels in plaintext on the LAN
- Any device on the same network can receive screenshots
- No device identity verification

### Recommended for Future Versions

| Feature | Approach |
|---|---|
| Encryption | TLS for TCP transfer, DTLS for UDP |
| Authentication | Pre-shared key or QR code pairing |
| Device Pinning | Allow/deny list by device ID |
| Content Signing | Ed25519 signature on screenshots |

## 9. Wireshark Filters

For debugging with Wireshark:

```
# All GrabDrop UDP traffic
udp.port == 9877

# Heartbeats only
udp.port == 9877 && frame contains "HEARTBEAT"

# Screenshot offers only
udp.port == 9877 && frame contains "SCREENSHOT_READY"

# TCP transfer (find the port from the offer)
tcp.port == 54321
```
