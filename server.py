from flask import Flask, jsonify, Response
import numpy as np
import cv2
import mss
import sounddevice as sd
import threading
import wave
import io
import time

app = Flask(__name__)

# SETTINGS
TARGET_MONITOR = 1

WIDTH = 200
HEIGHT = int(WIDTH * 9 / 16) # 16/9
FPS = 8
MODE = "Default" # Default, HighRes/Legacy/Groovy

AUDIO = True
AUDIO_BUFFER = 5
USE_MICROPHONE = False

last_sent_frame = None
force_full_frame = True
FRAME_BUFFER_SIZE = FPS
frame_buffer = []
buffer_lock = threading.Lock()

def quantize(img):
    # 16 levels per channel
    return (np.round(img / 17) * 17).astype(np.uint8)

def capture_frames():
    global frame_buffer

    with mss.mss() as sct:
        monitor = sct.monitors[TARGET_MONITOR]
        frame_interval = 1.0 / FPS

        while True:
            start_time = time.time()

            screenshot = np.array(sct.grab(monitor))[:, :, :3]
            small = cv2.resize(screenshot, (WIDTH, HEIGHT))
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            quantized = quantize(small)

            with buffer_lock:
                frame_buffer.append(quantized)
                if len(frame_buffer) > FRAME_BUFFER_SIZE:
                    frame_buffer.pop(0)

            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

@app.route("/frame")
def frame():
    global last_sent_frame, force_full_frame

    with buffer_lock:
        frames = frame_buffer.copy()

    updates_all = []

    if force_full_frame:
        reference_frame = None
        force_full_frame = False
    else:
        reference_frame = last_sent_frame if last_sent_frame is not None else frames[0] if frames else None

    for frame_img in frames:
        updates = []

        if reference_frame is None:
            ys, xs = np.indices((HEIGHT, WIDTH))
            ys = ys.flatten()
            xs = xs.flatten()

            for y, x in zip(ys, xs):
                r, g, b = frame_img[y, x]
                updates.append([int(x), int(y), int(r), int(g), int(b)])
        else:
            diff = np.any(frame_img != reference_frame, axis=2)
            ys, xs = np.where(diff)

            for y, x in zip(ys, xs):
                r, g, b = frame_img[y, x]
                updates.append([int(x), int(y), int(r), int(g), int(b)])

        updates_all.append(updates)
        reference_frame = frame_img.copy()

    if frames:
        last_sent_frame = frames[-1].copy()

    return jsonify({"frames": updates_all})

@app.route("/settings")
def settings():
    return jsonify({
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "audiobuffer": AUDIO_BUFFER,
        "mode": MODE,
        "audio": AUDIO
    })

@app.route("/reset")
def reset():
    global frame_buffer, last_sent_frame, force_full_frame
    with buffer_lock:
        frame_buffer = []
    last_sent_frame = None
    force_full_frame = True
    return "OK"

SAMPLE_RATE = 24000
CHANNELS = 1
BUFFER_SIZE = SAMPLE_RATE * (AUDIO_BUFFER + 1)

audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_index = 0
lock = threading.Lock()

def find_loopback():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "cable output" in dev["name"].lower() or USE_MICROPHONE and i == 0:
            print("Running with device "+dev["name"])
            return i
    return None

DEVICE = find_loopback()

def audio_callback(indata, frames, time_info, status):
    global buffer_index

    mono = indata.mean(axis=1)

    with lock:
        for sample in mono:
            audio_buffer[buffer_index] = sample
            buffer_index = (buffer_index + 1) % BUFFER_SIZE

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    device=DEVICE,
    callback=audio_callback,
)

stream.start()

@app.route("/audio.wav")
def get_audio():
    with lock:
        if buffer_index == 0:
            data = audio_buffer.copy()
        else:
            data = np.concatenate((
                audio_buffer[buffer_index:],
                audio_buffer[:buffer_index]
            ))

    pcm = np.int16(data * 32767)

    mem = io.BytesIO()
    with wave.open(mem, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    mem.seek(0)

    return Response(mem.read(), mimetype="audio/wav")

if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    app.run(host="0.0.0.0", port=5000)
