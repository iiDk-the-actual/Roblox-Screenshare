from flask import Flask, jsonify, Response
import numpy as np
import cv2
import mss
import sounddevice as sd
import threading
import wave
import io

app = Flask(__name__)

# SETTINGS
WIDTH = 200
HEIGHT = int(WIDTH * 9 / 16)
FPS = 8

AUDIO_BUFFER = 5
USE_MICROPHONE = False

prev_frame = None

def quantize(img):
    # 16 levels per channel
    return (np.round(img / 17) * 17).astype(np.uint8)

@app.route("/frame")
def frame():
    global prev_frame

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))

    screenshot = screenshot[:, :, :3]
    small = cv2.resize(screenshot, (WIDTH, HEIGHT))
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    quantized = quantize(small)

    updates = []

    if prev_frame is None:
        # First frame — send everything
        for y in range(HEIGHT):
            for x in range(WIDTH):
                r, g, b = quantized[y, x]
                updates.append([x, y, int(r), int(g), int(b)])
    else:
        diff = np.any(quantized != prev_frame, axis=2)
        ys, xs = np.where(diff)

        for y, x in zip(ys, xs):
            r, g, b = quantized[y, x]
            updates.append([int(x), int(y), int(r), int(g), int(b)])

    prev_frame = quantized.copy()

    return jsonify({"u": updates})

@app.route("/reset")
def reset():
    global prev_frame
    prev_frame = None
    return "OK"

SAMPLE_RATE = 24000
CHANNELS = 1
BUFFER_SIZE = SAMPLE_RATE * (AUDIO_BUFFER)

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
    app.run(host="0.0.0.0", port=5000)
