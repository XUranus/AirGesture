# GrabDrop Presentation Transcripts

**Total duration:** ~10 minutes (2.5 minutes per speaker)

**Format:** Each speaker section lists slide numbers and the spoken transcript.

---

## Speaker_A: Motivation & Overview (~2.5 min)

### Slide 1 — Title Slide
> Hello everyone, today we are going to present GrabDrop, a cross-device screenshot transfer system powered by air gesture recognition. This is a group project for DSAI5201, I will present the motivation and overview. After me, Speaker_B will cover the AI algorithm design, Speaker_C will talk about system architecture and implementation, and Speaker_D will present our results and future work.

### Slide 3 — The Problem
> So let us start with the problem. How do you currently share a screenshot between your phone and your laptop? You could use a chat app like WeChat — but that requires five or more steps: take the screenshot, open the app, select a contact, send, and then open it on the other device. You could use email or cloud storage, but those are even more cumbersome and require internet access. Even a USB cable requires physically connecting the devices. The key insight here is that all these methods require too many manual steps and interrupt your workflow.

### Slide 4 — Inspiration: Huawei Air Gesture
> Our project is inspired by Huawei's Air Gesture feature of Harmony NEXT, which lets you grab content from the screen using hand gestures and transfer it to another device. However, Huawei's implementation has major limitations — it only works between Huawei devices, it requires specific hardware, and it is not available on other Android brands or desktops. Our goal with GrabDrop is to remove these restrictions. We want to make this work on any Android device — we tested on Xiaomi — and bridge the gap between phones and laptops. Our solution is fully open-source, and it works entirely over the local network, so no server or internet is needed.

### Slide 5 — System Overview
> Here is how GrabDrop works at a high level. On Device A, the sender performs a GRAB gesture — that is, transitioning from an open palm to a closed fist in front of the camera. This triggers a screenshot capture and broadcasts the availability to all nearby devices on the same Wi-Fi network. Then, on Device B, the receiver performs a RELEASE gesture — transitioning from a fist to an open palm. This triggers a download of the screenshot from Device A.

### Slide 6 — Demo Flow
> To summarize the flow: both devices are running GrabDrop on the same Wi-Fi network. Step one: grab on Device A captures the screenshot and broadcasts it. Step two: release on Device B downloads and saves it. The total time is about three seconds. We support Android devices with Android 10 or higher, and desktop clients on any laptop with a front camera. Now I will hand over to Speaker_B to explain our AI algorithm design.

---

## Speaker_B: AI Algorithm Design (~2.5 min)

### Slide 8 — Two-Stage Pipeline
> Now I will explain the core AI algorithm behind our gesture detection. We designed a two-stage detection pipeline. The first stage is the IDLE stage, which runs at a low frame rate of about 10 frames per second. Its job is to detect the presence of a hand. When a hand is detected, the system transitions to the WAKEUP stage, which runs at 30 frames per second for high-precision motion tracking. This design balances power efficiency with detection accuracy — we do not waste battery running the camera at full speed all the time.

### Slide 9 — Hand Landmark Detection
> For hand detection, we use Google MediaPipe Hand Landmarker, a pre-trained model that outputs 21 three-dimensional landmarks on the hand. We use the float16 variant, which is about 12 megabytes in size. We run it in VIDEO mode, which enables stateful temporal tracking for more stable detections. We also lowered the detection confidence threshold from the default 0.5 to 0.3, which makes the system more sensitive in challenging lighting conditions.

### Slide 10 — Hand State Classification
> Given the 21 landmarks, we classify the hand state using a geometric metric we call the finger curl ratio. For each of the four fingers — index, middle, ring, and pinky — we compute the distance from the fingertip to the wrist, divided by the distance from the finger's base joint, called the MCP, to the wrist. If this ratio is greater than 1.3, the finger is extended. If it is less than 0.9, the finger is curled. We exclude the thumb because its movement axis is perpendicular to the other fingers, making this metric unreliable for it. If three or more fingers are extended, we classify the hand as PALM. If three or more are curled, it is a FIST.

### Slide 11 — Idle Stage
> In the idle stage, we maintain a sliding window of the last 10 frames. If 8 out of 10 frames show PALM, we enter the wakeup stage looking for a FIST, which would indicate a GRAB gesture. Similarly, if 8 out of 10 show FIST, we enter wakeup looking for PALM, which would be a RELEASE gesture. The idle stage uses only about 5 to 8 percent CPU, making it suitable for an always-on background service.

### Slide 12 — Wakeup Stage
> In the wakeup stage, we use consecutive-frame counting to confirm the gesture. The system counts how many frames in a row show the target state. If the target state is interrupted by even one frame of a different state, the counter resets to zero. We require 8 consecutive frames, which is about 267 milliseconds at 30 frames per second. If the gesture is not completed within 2 seconds, we time out and return to idle. After a confirmed gesture, there is a 3-second cooldown to prevent accidental re-triggering.

### Slide 13 — Time-Series CNN
> We also built a custom data capture tool using PyQt5 to record labeled gesture video clips. So far we have collected 42 clips covering grab, release, swipe-up, and swipe-down gestures. The plan is to train a lightweight 1D CNN on temporal sequences of landmark coordinates to replace the rule-based classification. The heuristic pipeline already works reliably, so the CNN is a planned enhancement for improved robustness. Now let me pass to Speaker_C for the system architecture.

---

## Speaker_C: System Architecture & Implementation (~2.5 min)

### Slide 15 — System Architecture
> Now I will explain the overall system architecture and implementation details. As you can see in this diagram, the system has several modular components. The camera feeds into the MediaPipe hand landmark detector, which outputs landmarks to the two-stage state machine. When the state machine emits a gesture event — either GRAB or RELEASE — the appropriate action is triggered. GRAB triggers the screen capture module, while RELEASE triggers the network manager to download a screenshot. We also have an overlay manager for visual feedback and a sound player for audio feedback.

### Slide 16 — Android Implementation
> On Android, the system runs as a foreground service, which is critical because it needs to operate independently of the activity lifecycle. The main orchestrator is GrabDropService. For gesture detection, we use CameraX ImageAnalysis, which is the modern Android camera API. One interesting engineering challenge was running CameraX inside a Service instead of an Activity — we solved this by implementing a custom ServiceLifecycleOwner. For screen capture, we use MediaProjection with a persistent VirtualDisplay that buffers the latest frame using an AtomicReference. The app requires 12 Android permissions, which we manage in a careful sequence. The Android app uses Jetpack Compose for its UI.

### Slide 17 — Desktop Implementation
> The desktop client is a modular Python application that mirrors the Android architecture. It uses OpenCV for camera access, the MediaPipe Python SDK for hand landmark detection, and the same two-stage state machine logic. For screen capture, it auto-detects the best backend for the user's desktop environment — cycling through tools like spectacle for KDE, grim for Wayland, gnome-screenshot, scrot, or falling back to the Python mss library. The key point is that both platforms share the exact same gesture detection logic and the same network protocol, which enables full cross-platform interoperability.

### Slide 18 — Network Protocol
> Our network protocol is zero-configuration — no server, no internet, no pairing required. It works in three phases. Phase one is device discovery using UDP heartbeats sent every 3 seconds to both a multicast address and a broadcast address on port 9877. When a device has not sent a heartbeat for 10 seconds, it is considered offline. Phase two is the screenshot offer — after a GRAB gesture captures a screenshot, the device starts a TCP server on a random port and broadcasts a SCREENSHOT_READY message with the port and file size. Phase three is the actual transfer over TCP — the receiver connects, sends a simple GET request, and receives a 4-byte length header followed by the PNG image data. We also handle retroactive matching — if the RELEASE gesture on the receiver happens before the UDP offer arrives, the system records the timestamp and automatically matches when the offer comes in within 3 seconds.

### Slide 19 — Swipe Detection
> As a bonus feature, we also detect vertical hand swipes for page navigation. During the wakeup stage, we track the hand center's Y position and look for at least 5 consecutive frames moving in the same direction with sufficient displacement. On Android, swipes are dispatched via AccessibilityService, and on desktop, they simulate PageUp and PageDown keys. The shorter cooldown of 0.8 seconds allows rapid consecutive swipes. Now I will pass to Speaker_D for results and future work.

---

## Speaker_D: Results, Limitations & Future Work (~2.5 min)

### Slide 21 — Performance Results
> Let me present our performance results. The end-to-end latency from grab to receive is approximately 2 to 4 seconds. This breaks down as follows: the idle stage takes about 1 second to trigger, since we need 8 out of 10 frames. The wakeup stage takes 0.3 to 1.5 seconds to confirm the gesture. And the network transfer over LAN takes 50 to 500 milliseconds depending on image size and network speed. In terms of resource usage, the idle stage uses only 5 to 8 percent CPU with the camera running at 10 frames per second. The wakeup stage uses 15 to 25 percent CPU, but it only lasts at most 2 seconds. This two-stage design keeps the battery impact minimal for always-on operation.

### Slide 22 — Strengths
> Our approach has several key strengths. First, it is cross-platform and cross-vendor — it works on any Android 10+ device and on Linux, macOS, and Windows desktops, with full interoperability between them. Second, it uses zero-configuration networking — no pairing, no cloud account, no internet required. We use dual multicast and broadcast for maximum LAN compatibility. Third, the power-efficient two-stage pipeline keeps idle CPU usage low. And fourth, the geometric finger-curl ratio is robust across different skin tones and lighting conditions, with consecutive-frame confirmation preventing false positives.

### Slide 23 — Limitations
> Of course, there are limitations. Low lighting conditions reduce MediaPipe's detection rate, which we partially address by lowering confidence thresholds. The system has no encryption — all data is plaintext on the LAN, which is acceptable for trusted networks but needs improvement. We only detect one hand at a time, and our gesture vocabulary is currently limited to grab, release, and swipe. The time-series CNN classifier is still in development, although the rule-based pipeline works reliably in practice.

### Slide 24 — Future Work
> Looking forward, we plan to train the time-series CNN classifier on our growing self-collected dataset to replace the rule-based heuristics and handle edge cases more robustly. We also plan to add security features like TLS encryption and device pairing via QR code. We are interested in extending the gesture vocabulary to include two-finger pinch, rotation, and custom user-defined gesture bindings. Other improvements include personalized calibration for different users, leveraging the 3D depth information from MediaPipe, and eventually building an iOS client.

### Slide 25 — Summary
> To summarize, we identified the problem that cross-device screenshot sharing requires too many manual steps. Inspired by Huawei Air Gesture but not limited by its proprietary ecosystem, we built GrabDrop — an open-source, cross-platform air gesture screenshot transfer system. It uses Google MediaPipe for hand landmark detection with our geometric finger-curl classifier, a two-stage detection pipeline for power efficiency, and a zero-configuration LAN protocol for transfer. It works on any Android device and Linux, macOS, and Windows desktops, achieving end-to-end transfer in about 3 seconds with minimal battery impact. All our code is open source. Thank you for listening, and we are happy to take questions.
