# pip install faster-whisper ultralytics sounddevice scipy opencv-python fuzzywuzzy gTTS playsound==1.2.2 ctranslate2

import os
import time
import math
import tempfile
import subprocess
import json
from datetime import datetime
import numpy as np
import cv2
import threading
import requests
import asyncio
import sounddevice as sd
import scipy.io.wavfile as wav
import torch
from faster_whisper import WhisperModel
from ultralytics import YOLO
from fuzzywuzzy import fuzz
from gtts import gTTS
from playsound import playsound
from typing import Optional

# --- API server imports ---
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except Exception:
    FastAPI = None  # type: ignore
    WebSocket = None  # type: ignore
    WebSocketDisconnect = None  # type: ignore
    CORSMiddleware = None  # type: ignore
    uvicorn = None  # type: ignore

# ---------------------- CONFIG ----------------------
# Video source selection:
# 1) If CAM_SOURCE is a non-empty string, it will be used directly (e.g., "http://<esp32-ip>:81/stream").
# 2) Otherwise, the script will search for a local webcam index.
CAM_SOURCE = os.getenv("ESP32_CAM_URL",
                       "http://10.128.72.247:81/stream").strip()  # e.g., "http://192.168.1.50:81/stream"

MIC_DEVICE_INDEX = 1
ASR_MODEL_SIZE = "small"
YOLO_MODEL_PATH = "yolov8m.pt"  # replace with your custom best.pt when ready
SAMPLE_RATE = 16000
LISTEN_SECONDS = 3
POST_SPEAK_DELAY = 0.5
PROMPT_BEEP = True
SCAN_SECONDS = 20

# YOLO inference knobs
DETECT_CONF = 0.55
DETECT_IOU = 0.5
DETECT_IMGSZ = 640
MAX_DET = 100
AGNOSTIC_NMS = False

# Smoothing/visibility
SHOW = True
WINDOW_NAME = "Stable Room Scan"
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480  # Fixed window size

# Counting stability
MIN_HITS_TO_COUNT = 3
STALE_FRAMES = 30

# TTS
TTS_LANG = "en"
TTS_RATE = 1.4
TTS_SLOW = None

# ---------------------- JSON persistence ----------------------
RESULT_JSON_PATH = os.path.abspath("F:/iotproject/output/room_scan_results.json")

# ---------------------- Runtime state for API ----------------------
ESP32_IP: Optional[str] = None
FLASHLIGHT_MODE: str = "auto"  # one of: auto, on, off
MOTION_ENABLED: bool = False
SCAN_THREAD: Optional[threading.Thread] = None
SCAN_RUNNING: bool = False
LAST_RESULT_TEXT: str = ""
_flash_last_state: Optional[bool] = None
_flash_last_time: float = 0.0
_flash_cooldown_s: float = 0.5

# WebSocket client set
WS_CLIENTS_LOCK = threading.Lock()
WS_CLIENTS: set = set()


def append_scan_result(entry: dict, path: str = RESULT_JSON_PATH):
    """Appends a scan result entry to the JSON file, creating the directory if needed."""
    try:
        # --- FIX: Create the output directory if it doesn't exist ---
        dir_path = os.path.dirname(path)
        if dir_path:  # Check if path includes a directory
            os.makedirs(dir_path, exist_ok=True)
        # --- End of Fix ---

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or "history" not in data:
                    data = {"history": []}
            except json.JSONDecodeError:
                print(f"⚠ Warning: JSON file at {path} is corrupted. Creating new history.")
                data = {"history": []}
        else:
            data = {"history": []}

        data["history"].append(entry)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"📝 Saved scan result to {path}")
    except Exception as e:
        print(f"❌ Failed to write JSON to {path}: {e}")


# ---------------------- Light ON/OFF via brightness ----------------------
BRIGHTNESS_THRESHOLD = 60.0  # 0-255 scale; tweak as needed
BRIGHTNESS_TEXT_POS = (10, 30)  # top-left
BRIGHTNESS_FONT = cv2.FONT_HERSHEY_SIMPLEX
BRIGHTNESS_FONT_SCALE = 0.8
BRIGHTNESS_THICKNESS = 2

# ---------------------- Voice phrases ----------------------
PHRASES = [
    "scan room", "room scan", "scan the room", "look around the room",
    "what's in this room", "detect room objects", "inventory the room",
    "scan area", "look around", "detect objects", "analyze room", "room detection",
    "check what's in my room", "dekho room mein kya hai", "scan karo room ko",
    "show me what’s here", "room ka scan chalu karo", "tell me what you see",
    "detect items in this room", "list what’s in this room", "scan this area",
    "check the things around", "find all objects here", "find things in the room",
    "scan karo yahan", "look what’s around me", "analyze this room",
    "room mein kya kya cheezein hain", "start room scanning",
    "see what’s in front", "check what’s visible", "kya hai room mein", "mujhe room dikhayo"
]

# ---------------------- Labels and canonical mapping ----------------------
PERSON_ALIASES = {"person", "man", "woman", "boy", "girl", "kid", "child"}

INDOOR_NAMES = {
    *PERSON_ALIASES,
    "chair", "stool", "bench", "sofa", "couch", "bed", "mattress",
    "dining table", "table", "desk", "wardrobe", "almirah", "cupboard",
    "shelf", "rack", "drawer", "side table", "study table",
    "tv", "television", "monitor", "laptop", "computer", "keyboard", "mouse",
    "remote", "mobile", "cell phone", "charger", "router", "speaker",
    "fan", "ceiling fan", "table fan", "ac", "air conditioner", "cooler",
    "fridge", "refrigerator", "washing machine", "microwave", "oven",
    "geyser", "heater", "camera", "bulb", "tube light", "lamp", "light",
    "bottle", "cup", "glass", "mug", "plate", "bowl", "pan", "pot",
    "kettle", "pressure cooker", "induction", "gas stove", "utensil",
    "toilet", "sink", "shower", "towel", "mirror", "bucket", "soap",
    "curtain", "pillow", "blanket", "bedsheet", "mat", "carpet", "rug",
    "painting", "photo frame", "vase", "potted plant", "flower pot", "clock",
    "backpack", "bag", "luggage", "box", "container", "hanger",
    "book", "notebook", "pen", "newspaper", "magazine", "toy", "teddy bear",
    "switchboard", "plug", "extension board",
    "tvmonitor", "diningtable"
}

CANONICAL_MAP = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person", "kid": "person", "child": "person",
    "tvmonitor": "tv", "television": "tv",
    "diningtable": "dining table",
    "mobile": "cell phone", "mobile phone": "cell phone",
    "couch": "sofa",
    "fridge": "refrigerator",
    "microwave oven": "microwave",
    "flower pot": "potted plant"
}

FURNITURE_SET = {n for n in INDOOR_NAMES if n not in PERSON_ALIASES}

# ---------------------- Device and model init ----------------------
CUDA = torch.cuda.is_available()
MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
FW_DEVICE = "cuda" if CUDA else "cpu"
YOLO_DEVICE = 0 if CUDA else ("mps" if MPS else "cpu")
YOLO_HALF = bool(CUDA)


def init_asr(model_size: str):
    """Initializes the Faster-Whisper model, trying different compute types."""
    if FW_DEVICE == "cuda":
        candidates = ["float16", "int8", "float32"]
    else:
        candidates = ["int8", "float32"]
    last_err = None
    for ct in candidates:
        try:
            print(f"Loading Faster-Whisper on {FW_DEVICE} with compute_type={ct} ...")
            m = WhisperModel(model_size, device=FW_DEVICE, compute_type=ct)
            print(f"ASR ready: compute_type={ct}")
            return m, ct
        except Exception as e:
            print(f"ASR init failed for compute_type={ct}: {e}")
            last_err = e
    raise last_err if last_err else RuntimeError("ASR initialization failed")


print("Initializing ASR (Whisper)...")
asr_model, FW_COMPUTE_TYPE = init_asr(ASR_MODEL_SIZE)

print("Loading YOLO model...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load YOLO model from {YOLO_MODEL_PATH}")
    print(f"Error: {e}")
    print("Please make sure the model file exists and all dependencies are installed.")
    exit(1)

idx_to_name = yolo_model.names
name_to_idx = {v: k for k, v in idx_to_name.items()}
# Get all class indices that match our indoor names or are aliases
CLASSES_ALL = [name_to_idx[name] for name in idx_to_name.values()
               if (name in INDOOR_NAMES) or (name in CANONICAL_MAP)]
if not CLASSES_ALL:
    print("⚠ Warning: No matching classes found between YOLO model and INDOOR_NAMES.")
    print("The model might not detect anything unless INDOOR_NAMES or CANONICAL_MAP are updated.")


# ---------------------- TTS ----------------------
def _apply_tempo_with_ffmpeg(src_mp3: str, rate: float) -> str:
    """Uses ffmpeg to change audio speed and returns path to new temp file."""
    stages, r = [], rate
    while r > 2.0:
        stages.append(2.0);
        r /= 2.0
    while r < 0.5:
        stages.append(0.5);
        r /= 0.5
    stages.append(r)
    atempo = ",".join(f"atempo={x:.6g}" for x in stages)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf_out:
        dst = tf_out.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_mp3, "-filter:a", atempo, "-vn", dst],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return dst
    except FileNotFoundError:
        print("⚠ Warning: 'ffmpeg' not found in PATH. Cannot change TTS speed.")
        raise  # Re-raise to be caught by 'speak' function
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: ffmpeg command failed: {e}")
        raise  # Re-raise to be caught by 'speak' function


def speak(text: str, lang: str = None, rate: float = None, slow: bool | None = None):
    """Speaks the given text using gTTS and playsound."""
    lang = TTS_LANG if lang is None else lang
    rate = TTS_RATE if rate is None else rate
    slow = TTS_SLOW if slow is None else slow
    print("💬 AI:", text)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf_in:
            mp3_in = tf_in.name
        gTTS(text=text, lang=lang, slow=(False if slow is None else slow)).save(mp3_in)
        mp3_to_play = mp3_in
    except Exception as e:
        print(f"❌ Failed to generate TTS audio: {e}")
        return

    try:
        if abs(rate - 1.0) > 1e-3:
            try:
                mp3_to_play = _apply_tempo_with_ffmpeg(mp3_in, rate)
            except Exception as ex:
                print(f"FFmpeg tempo adjust failed; playing at normal speed: {ex}")
                mp3_to_play = mp3_in
        playsound(mp3_to_play)
    except Exception as e:
        print(f"❌ Failed to play sound: {e}")
        print("If this error persists, try 'pip install playsound==1.2.2'")
    finally:
        # Clean up temporary files
        for p in {mp3_in, mp3_to_play}:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"⚠ Warning: Could not remove temp file {p}: {e}")
    time.sleep(POST_SPEAK_DELAY)


# ---------------------- Audio / ASR ----------------------
def record_audio(filename="input.wav", duration=LISTEN_SECONDS, fs=SAMPLE_RATE):
    """Records audio from the specified mic index."""
    print("🎤 Recording...")
    try:
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16", device=MIC_DEVICE_INDEX)
        sd.wait()
        if np.all(rec == 0):
            print("⚠ Warning: Recording is silent. Check mic/device index.")
        wav.write(filename, fs, rec)
        print("✅ Saved:", filename)
        return True
    except Exception as e:
        print(f"❌ Failed to record audio: {e}")
        print(f"Check if MIC_DEVICE_INDEX={MIC_DEVICE_INDEX} is correct.")
        return False


def listen(duration=LISTEN_SECONDS):
    """Records and transcribes audio."""
    if not record_audio("input.wav", duration=duration):
        return ""  # Return empty string if recording failed

    try:
        segments, _ = asr_model.transcribe("input.wav", vad_filter=True, beam_size=1)
        text = "".join(seg.text for seg in segments).strip().lower()
        print("🗣 You said:", text)
        return text
    except Exception as e:
        print(f"❌ Failed to transcribe audio: {e}")
        return ""


# ---------------------- Camera Helpers ----------------------
def open_camera(index=0):
    """Tries to open a camera index and returns it if successful."""
    if os.name == 'nt':
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap.release()
        return None

    ok, _ = cap.read()
    cap.release()
    return index if ok else None


def find_cam(max_index=5):
    """Finds the first available local webcam index."""
    print("Searching for local webcam...")
    for i in range(max_index + 1):
        idx = open_camera(i)
        if idx is not None:
            print(f"Found webcam at index {idx}")
            return idx
    print("No local webcam found.")
    return None


# ---------------------- Smoothing ----------------------
class LowPass:
    """Simple low-pass filter."""

    def __init__(self):
        self.y = None

    def filter(self, x, a):
        if self.y is None:
            self.y = x
        else:
            self.y = a * x + (1 - a) * self.y
        return self.y


def _alpha(fc, dt):
    """Calculates smoothing factor alpha from cutoff frequency and time delta."""
    r = 2 * math.pi * fc * dt
    return r / (r + 1.0)


class OneEuro:
    """1-Euro filter for smoothing noisy signals."""

    def __init__(self, min_cutoff=1.0, beta=0.015, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_f = LowPass()
        self.dx_f = LowPass()
        self.last_t = None
        self.last_x = None

    def __call__(self, x, t):
        if self.last_t is None:
            dt = 1.0 / 30.0  # Assume 30fps for first frame
        else:
            dt = max(1e-3, t - self.last_t)
        self.last_t = t

        dx = 0.0 if self.last_x is None else (x - self.last_x) / dt
        self.last_x = x

        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = self.dx_f.filter(dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        return self.x_f.filter(x, a)


def get_track_filters(filters_dict, tid):
    """Gets or creates a set of 1-Euro filters for a given track ID."""
    if tid not in filters_dict:
        filters_dict[tid] = (OneEuro(), OneEuro(), OneEuro(), OneEuro())
    return filters_dict[tid]


# ---------------------- Detection / tracking ----------------------
def detect_room(duration=SCAN_SECONDS, source: str | int | None = None):
    """Starts the YOLOv8 tracking and room scanning process."""
    # Decide the video source: explicit arg > env var CAM_SOURCE > local webcam
    if source is not None:
        src = source
    elif CAM_SOURCE:
        src = CAM_SOURCE  # ESP32-CAM MJPEG stream, e.g., "http://<ip>:81/stream"
        print(f"Using camera source from CAM_SOURCE: {src}")
    else:
        cam_index = find_cam(5)
        if cam_index is None:
            speak("No camera found. Close other apps using the camera and try again.")
            return
        src = cam_index

    # Initialize tracking from URL or camera index
    try:
        results_gen = yolo_model.track(
            source=src,
            stream=True,
            persist=True,
            tracker="botsort.yaml",
            device=YOLO_DEVICE,
            half=YOLO_HALF,
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            imgsz=DETECT_IMGSZ,
            max_det=MAX_DET,
            classes=CLASSES_ALL if CLASSES_ALL else None,
            agnostic_nms=AGNOSTIC_NMS,
            verbose=False,
            show=False
        )
    except Exception as e:
        print(f"❌ Failed to start YOLO tracking on source: {src}")
        print(f"Error: {e}")
        speak("I couldn't access the camera feed. Please check the connection or source.")
        return

    if SHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    start = time.time()
    frame_id = 0

    box_filters = {}
    track_last_seen = {}
    person_ids_current = set()
    peak_persons = 0
    furniture_hits = {name: {} for name in FURNITURE_SET}

    # Brightness tracking
    brightness_filter = OneEuro(min_cutoff=0.2, beta=0.01, d_cutoff=1.0)
    brightness_values = []
    light_on = False

    speak(f"Scanning the room for {duration} seconds, please pan the camera slowly.")
    _notify_ws({"type": "scan_started", "duration": duration})

    try:
        for r in results_gen:
            if time.time() - start > duration:
                break

            frame = r.orig_img.copy()
            now = time.time()
            person_ids_current.clear()

            # Brightness and overlay
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2]
                raw_brightness = float(np.mean(v))
                filt_brightness = float(brightness_filter(raw_brightness, now))
                brightness_values.append(filt_brightness)
                light_on = bool(filt_brightness < BRIGHTNESS_THRESHOLD)
                _maybe_set_flashlight(light_on)
                light_text = f"Light: {'ON' if light_on else 'OFF'} ({int(filt_brightness)})"
                color = (0, 0, 255) if light_on else (0, 200, 0)
                cv2.putText(frame, light_text, BRIGHTNESS_TEXT_POS, BRIGHTNESS_FONT,
                            BRIGHTNESS_FONT_SCALE, color, BRIGHTNESS_THICKNESS, cv2.LINE_AA)
            except Exception:
                pass  # Ignore brightness calculation errors on bad frames

            # Detections / tracking
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                ids = getattr(r.boxes, "id", None)
                clss = r.boxes.cls.int().cpu().tolist() if r.boxes.cls is not None else [None] * len(xyxy)
                confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes.conf is not None else [None] * len(xyxy)
                names = r.names

                for i, bb in enumerate(xyxy):
                    x1, y1, x2, y2 = bb.tolist()
                    tid = int(ids[i].item()) if ids is not None and ids[i] is not None else -1
                    cls_id = clss[i]
                    conf = confs[i] if confs and confs[i] is not None else 0.0

                    if cls_id is None: continue  # Skip if class ID is missing

                    raw_label = names[int(cls_id)]
                    canonical = CANONICAL_MAP.get(raw_label, raw_label)

                    if conf < 0.6:
                        continue

                    if tid >= 0:
                        f_x1, f_y1, f_x2, f_y2 = get_track_filters(box_filters, tid)
                        sx1 = f_x1(x1, now)
                        sy1 = f_y1(y1, now)
                        sx2 = f_x2(x2, now)
                        sy2 = f_y2(y2, now)
                        track_last_seen[tid] = frame_id
                    else:
                        sx1, sy1, sx2, sy2 = x1, y1, x2, y2

                    if canonical == "person" and tid >= 0:
                        person_ids_current.add(tid)
                        if MOTION_ENABLED:
                            _notify_ws({
                                "type": "motion",
                                "label": "person",
                                "track_id": tid,
                                "conf": round(conf, 3)
                            })

                    if canonical in FURNITURE_SET and tid >= 0:
                        hits = furniture_hits[canonical].get(tid, 0) + 1
                        furniture_hits[canonical][tid] = hits

                    p1 = (int(round(sx1)), int(round(sy1)))
                    p2 = (int(round(sx2)), int(round(sy2)))
                    color = (0, 255, 0) if canonical == "person" else (0, 128, 255)
                    cv2.rectangle(frame, p1, p2, color, 2)
                    txt = f"{canonical}#{tid if tid >= 0 else '-'} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), color, -1)
                    cv2.putText(frame, txt, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                                cv2.LINE_AA)

            peak_persons = max(peak_persons, len(person_ids_current))

            # Prune stale trackers
            stale_cut = frame_id - STALE_FRAMES
            if stale_cut > 0:
                to_del = [tid for tid, last in track_last_seen.items() if last < stale_cut]
                for tid in to_del:
                    box_filters.pop(tid, None)
                    track_last_seen.pop(tid, None)

            if SHOW:
                try:
                    # Resize to fixed window size for consistent display
                    disp = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    cv2.imshow(WINDOW_NAME, disp)
                except Exception:
                    # Fallback if resize fails
                    cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Scan interrupted by user.")
                    break

            frame_id += 1

    except Exception as e:
        print(f"❌ An error occurred during the tracking loop: {e}")
        speak("An error occurred during the scan.")

    finally:
        if SHOW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # --- Post-scan summary ---

    # Compute stable furniture counts
    furn_counts = {}
    for cname, tids in furniture_hits.items():
        stable = sum(1 for _tid, hits in tids.items() if hits >= MIN_HITS_TO_COUNT)
        if stable > 0:
            furn_counts[cname] = stable

    # Occupancy text
    if peak_persons <= 0:
        occ_text = "There are no people"
    elif peak_persons == 1:
        occ_text = "There is one person"
    else:
        occ_text = f"There are {peak_persons} people"

    # Furniture text
    if furn_counts:
        parts = [f"{n} {k if n == 1 else k + 's'}" for k, n in sorted(furn_counts.items())]
        furn_text = ", and I see " + ", ".join(parts)
    else:
        furn_text = ", and I don't see any noteworthy furniture"

    # Brightness summary and light status
    avg_brightness = float(np.mean(brightness_values)) if brightness_values else None
    # Use last known brightness value for final light state
    final_light_on = bool(
        (brightness_values[-1] if brightness_values else BRIGHTNESS_THRESHOLD + 1) < BRIGHTNESS_THRESHOLD)
    light_status_text = "ON" if final_light_on else "OFF"
    light_summary_text = f"The light appears to be {light_status_text}."

    # Final spoken summary
    summary_text = f"Room scan complete. {occ_text}{furn_text}. {light_summary_text}"
    speak(summary_text)
    global LAST_RESULT_TEXT
    LAST_RESULT_TEXT = summary_text
    _notify_ws({"type": "scan_complete", "summary": summary_text, "persons": peak_persons, "furniture": furn_counts})

    # Persist to JSON
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "peak_persons": peak_persons,
        "furniture_counts": furn_counts,
        "summary": summary_text,
        "brightness_avg": avg_brightness,
        "brightness_threshold": BRIGHTNESS_THRESHOLD,
        "light_state": light_status_text
    }
    append_scan_result(entry)


# ---------------------- ESP32 Flashlight control ----------------------
def _esp32_base_url() -> Optional[str]:
    return f"http://{ESP32_IP}" if ESP32_IP else None


def _send_esp32_flash(on: bool) -> bool:
    base = _esp32_base_url()
    if not base:
        return False
    paths = [
        (f"{base}/control?var=led_intensity&val={'255' if on else '0'}"),
        (f"{base}/control?var=flash&val={'1' if on else '0'}"),
        (f"{base}/led?state={'1' if on else '0'}"),
    ]
    for url in paths:
        try:
            r = requests.get(url, timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False


def _maybe_set_flashlight(should_turn_on: bool):
    global _flash_last_state, _flash_last_time
    # Apply mode
    desired_on = should_turn_on if FLASHLIGHT_MODE == "auto" else (FLASHLIGHT_MODE == "on")
    now = time.time()
    if _flash_last_state is not None and _flash_last_state == desired_on and (now - _flash_last_time) < _flash_cooldown_s:
        return
    if _send_esp32_flash(desired_on):
        _flash_last_state = desired_on
        _flash_last_time = now


# ---------------------- WebSocket helpers ----------------------
def _notify_ws(payload: dict):
    if not WS_CLIENTS:
        return
    dead = []
    with WS_CLIENTS_LOCK:
        for ws in list(WS_CLIENTS):
            try:
                # Schedule send in thread-safe way
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(ws.send_json(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            WS_CLIENTS.discard(ws)


# ---------------------- FastAPI Server ----------------------
def _create_app():
    if FastAPI is None:
        return None
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/connect")
    def connect(payload: dict):
        global ESP32_IP, CAM_SOURCE
        ip = payload.get("ip", "").strip()
        if not ip:
            return {"ok": False, "error": "ip required"}
        ESP32_IP = ip
        CAM_SOURCE = f"http://{ip}:81/stream"
        return {"ok": True, "cam_source": CAM_SOURCE}

    @app.post("/flashlight")
    def set_flashlight(payload: dict):
        global FLASHLIGHT_MODE
        mode = payload.get("mode", "").strip().lower()
        if mode not in {"auto", "on", "off"}:
            return {"ok": False, "error": "mode must be auto|on|off"}
        FLASHLIGHT_MODE = mode
        # Apply immediately based on last state
        _maybe_set_flashlight(True)
        return {"ok": True, "mode": FLASHLIGHT_MODE}

    @app.post("/motion")
    def toggle_motion(payload: dict):
        global MOTION_ENABLED
        enabled = bool(payload.get("enabled", False))
        MOTION_ENABLED = enabled
        return {"ok": True, "enabled": MOTION_ENABLED}

    @app.post("/scan")
    def scan(payload: dict):
        global SCAN_THREAD, SCAN_RUNNING
        start = bool(payload.get("start", True))
        duration = int(payload.get("duration", SCAN_SECONDS))
        if start:
            if SCAN_RUNNING:
                return {"ok": False, "error": "scan already running"}
            def _run():
                global SCAN_RUNNING
                SCAN_RUNNING = True
                try:
                    detect_room(duration=duration)
                finally:
                    SCAN_RUNNING = False
            SCAN_THREAD = threading.Thread(target=_run, daemon=True)
            SCAN_THREAD.start()
            return {"ok": True, "started": True, "duration": duration}
        else:
            # Best-effort: stopping is cooperative via 'q' or timeout; not implemented forcefully
            return {"ok": False, "error": "stop not implemented; wait for timeout"}

    @app.get("/status")
    def status():
        return {
            "ok": True,
            "esp32_ip": ESP32_IP,
            "flashlight_mode": FLASHLIGHT_MODE,
            "motion_enabled": MOTION_ENABLED,
            "scan_running": SCAN_RUNNING,
            "last_result": LAST_RESULT_TEXT,
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        with WS_CLIENTS_LOCK:
            WS_CLIENTS.add(ws)
        try:
            await ws.send_json({"type": "hello", "status": "connected"})
            while True:
                try:
                    await ws.receive_text()  # keepalive or ignore client messages
                except Exception:
                    await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass
        finally:
            with WS_CLIENTS_LOCK:
                WS_CLIENTS.discard(ws)

    return app


# ---------------------- Commands / Main ----------------------
def is_room_scan_command(command: str) -> bool:
    """Checks if the command is a room scan trigger phrase."""
    for p in PHRASES:
        if fuzz.partial_ratio(command, p) > 70:
            return True
    return False


def prompt_and_listen(prompt_text="Say a command after the beep.", duration=LISTEN_SECONDS):
    """Gives a spoken prompt, plays a beep, and listens."""
    speak(prompt_text)
    if PROMPT_BEEP:
        try:
            if os.name == 'nt':
                import winsound
                winsound.Beep(880, 120)
            else:
                # Simple beep for non-Windows
                print("\a", end="", flush=True)
        except Exception:
            pass  # Ignore beep errors
        time.sleep(0.2)
    return listen(duration=duration)


if __name__ == "__main__":
    # If FastAPI is available, serve the control API; else fall back to voice loop
    app = _create_app()
    if app is not None and uvicorn is not None:
        print("Starting API server on http://0.0.0.0:8000 ...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            _ = cv2.getWindowProperty if hasattr(cv2, "getWindowProperty") else None
        except Exception:
            print("Warning: Could not access OpenCV GUI properties. Display might not work.")
        speak("System ready.")
        while True:
            cmd = prompt_and_listen("Say scan room to start, or say stop.", duration=LISTEN_SECONDS)
            if not cmd:
                continue
            if "stop" in cmd or "exit" in cmd or "quit" in cmd:
                speak("Stopping program. Goodbye!")
                break
            if is_room_scan_command(cmd):
                detect_room()
            else:
                speak("That was not a room scan command.")