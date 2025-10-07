from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2, mediapipe as mp, numpy as np, json, pickle, os, threading, time
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -------------------------------
# Camera manager (singleton)
# -------------------------------
class CameraManager:
    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.raw_frame = None
        self.running = False
        self.owner = None
        self.thread = None
        self.should_stop = False

    def _open_capture(self):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            try:
                cap = cv2.VideoCapture(self.index, backend)
                if cap and cap.isOpened():
                    print(f"Camera opened with backend={backend}")
                    return cap
                if cap: cap.release()
            except Exception:
                pass
        raise RuntimeError("No camera backend available or camera busy")

    def start(self, owner):
        with self.lock:
            if self.owner and self.owner != owner:
                return False, f"Camera in use by {self.owner}"
            if self.running:
                self.owner = owner
                self.latest_frame = None
                self.raw_frame = None
                return True, "Already running"
            self.cap = self._open_capture()
            self.should_stop = False
            self.owner = owner
            self.running = True
            self.latest_frame = None
            self.raw_frame = None
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            return True, "Camera started"

    def stop(self, owner):
        with self.lock:
            if self.owner != owner and self.owner is not None:
                return False, f"Camera owned by {self.owner}"
            self.should_stop = True
        if self.thread:
            self.thread.join(timeout=2)
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap = None
            self.running = False
            self.owner = None
            self.latest_frame = None
            self.raw_frame = None
            self.thread = None
        return True, "Camera stopped"

    def _reader_loop(self):
        try:
            while not self.should_stop:
                if not self.cap:
                    time.sleep(0.02)
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
                self.raw_frame = frame
                _, buffer = cv2.imencode('.jpg', frame)
                self.latest_frame = buffer.tobytes()
                time.sleep(0.02)
        except Exception as e:
            print("Camera reader loop error:", e)
        finally:
            if self.cap:
                self.cap.release()

    def get_jpeg(self):
        return self.latest_frame

    def get_raw(self):
        return self.raw_frame

camera_mgr = CameraManager(index=0)

# -------------------------------
# Load models / classes
# -------------------------------
with open('./classes/ALPH_CLASSES.json', 'r') as f:
    alph_labels = {int(k): v for k, v in json.load(f).items()}
with open('./classes/NUM_CLASSES.json', 'r') as f:
    num_labels = {int(k): v for k, v in json.load(f).items()}

alph_model_dict = pickle.load(open('./models/alph_model.p', 'rb'))
num_model_dict = pickle.load(open('./models/num_model.p', 'rb'))

lstm_model = load_model("./models/lstm_phrase_model.h5")
label_classes = ["kumustaka", "no_hand", "non_sign"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -------------------------------
# Globals
# -------------------------------
current_model = None
current_labels = None
current_model_type = None
last_realtime_prediction = {"prediction": None, "confidence": None}

# -------------------------------
# Utilities
# -------------------------------
def extract_landmarks_from_bgr(frame, hands_solver, include_z=True):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_solver.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None
    feats = []
    for hand_landmarks in results.multi_hand_landmarks[:1]:
        for lm in hand_landmarks.landmark:
            feats += [lm.x, lm.y] + ([lm.z] if include_z else [])
    return feats

def prepare_2d_input_from_landmarks(flat_xy):
    if not flat_xy or len(flat_xy) % 2 != 0:
        return None
    xs, ys = flat_xy[0::2], flat_xy[1::2]
    minx, miny = min(xs), min(ys)
    data_aux = [(x - minx) for x in xs for _ in (0, 1)]
    for i, y in enumerate(ys):
        data_aux[2*i + 1] = y - miny
    return data_aux

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera/<mode>', methods=['POST'])
def start_camera(mode):
    if mode not in ('live', 'realtime'):
        return jsonify({"error": "invalid mode"}), 400
    ok, msg = camera_mgr.start(mode)
    return jsonify({"success": ok, "message": msg})

@app.route('/stop_camera/<mode>', methods=['POST'])
def stop_camera(mode):
    if mode not in ('live', 'realtime'):
        return jsonify({"error": "invalid mode"}), 400
    ok, msg = camera_mgr.stop(mode)
    return jsonify({"success": ok, "message": msg})

@app.route('/load_model/<model_type>')
def load_model_route(model_type):
    global current_model, current_labels, current_model_type
    try:
        if model_type == 'alphabet':
            current_model = alph_model_dict['model']
            current_labels = alph_labels
        elif model_type == 'number':
            current_model = num_model_dict['model']
            current_labels = num_labels
        else:
            return jsonify(success=False), 400
        current_model_type = model_type
        return jsonify(success=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# Live Stream
# -------------------------------
def live_stream_generator():
    while True:
        if not camera_mgr.running or camera_mgr.owner != 'live':
            time.sleep(0.05)
            continue
        frame = camera_mgr.get_raw()
        if frame is None:
            time.sleep(0.02)
            continue
        out = frame.copy()
        try:
            with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5) as hands:
                feats = extract_landmarks_from_bgr(out, hands, include_z=False)
                if feats and current_model:
                    vec = prepare_2d_input_from_landmarks(feats)
                    pred = current_model.predict([np.asarray(vec)])
                    label = current_labels[int(pred[0])]
                    xs, ys = feats[0::2], feats[1::2]
                    H, W, _ = out.shape
                    x1, y1, x2, y2 = int(min(xs)*W)-10, int(min(ys)*H)-10, int(max(xs)*W)+10, int(max(ys)*H)+10
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(out, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(out, lm, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())
        except Exception as e:
            print("Live frame error:", e)
        _, buf = cv2.imencode('.jpg', out)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(live_stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------
# Realtime Stream
# -------------------------------
def realtime_stream_generator():
    with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2) as hands:
        window = []
        while True:
            if not camera_mgr.running or camera_mgr.owner != 'realtime':
                time.sleep(0.05)
                continue
            frame = camera_mgr.get_raw()
            if frame is None:
                time.sleep(0.02)
                continue
            out = frame.copy()
            results = hands.process(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            feats = []
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks[:2]:
                    mp_drawing.draw_landmarks(out, hand, mp_hands.HAND_CONNECTIONS)
                    for lm in hand.landmark:
                        feats += [lm.x, lm.y, lm.z]
            if feats:
                feats = (feats + [0.0]*126)[:126]
                window.append(feats)
                if len(window) > 70: window.pop(0)
            if len(window) == 70:
                seq = np.expand_dims(np.array(window), axis=0)
                preds = lstm_model.predict(seq, verbose=0)
                idx, conf = int(np.argmax(preds)), float(np.max(preds))
                lbl = label_classes[idx]
                last_realtime_prediction.update({"prediction": lbl, "confidence": conf})
                cv2.putText(out, f"{lbl} {conf:.2f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            _, buf = cv2.imencode('.jpg', out)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.03)

@app.route('/realtime_feed')
def realtime_feed():
    return Response(realtime_stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime_result')
def realtime_result():
    return jsonify(last_realtime_prediction)

# -------------------------------
# FSL asset mapping
# -------------------------------
@app.route('/fsl_map')
def fsl_map():
    base_videos = os.path.join(app.static_folder, 'fsl_videos')
    base_letters = os.path.join(app.static_folder, 'fsl_letters')
    mapping = {}
    # letters
    for ch in [chr(i) for i in range(ord('a'), ord('z')+1)]:
        fn = f"{ch}.jpg"
        if os.path.exists(os.path.join(base_letters, fn)):
            mapping[ch] = fn
    # phrases
    if os.path.exists(os.path.join(base_videos, "kumusta_ka.mp4")):
        mapping["kumusta ka"] = "kumusta_ka.mp4"
    return jsonify(mapping)

@app.route('/fsl_video/<path:filename>')
def fsl_video(filename):
    return send_from_directory(os.path.join(app.static_folder, 'fsl_videos'), filename)

@app.route('/fsl_letter/<path:filename>')
def fsl_letter(filename):
    return send_from_directory(os.path.join(app.static_folder, 'fsl_letters'), filename)

# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


