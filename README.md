# Gaze Attention Detector

**Reveal an image when you look away from the screen.**  
A small computer-vision project using webcam + face landmarks to detect attention and trigger an action (show full image) when gaze leaves the screen.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Webcam    │────▶│  MediaPipe       │────▶│  Gaze logic         │
│   (OpenCV)  │     │  Face Mesh       │     │  nose offset vs     │
└─────────────┘     │  468 landmarks   │     │  face center        │
                    └──────────────────┘     └──────────┬──────────┘
                                                        │
                    ┌──────────────────┐                │
                    │  Trigger logic   │◀────────────────┘
                    │  N frames away  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Show full image │
                    │  (reveal window) │
                    └──────────────────┘
```

- **Capture**: OpenCV reads from the default webcam.
- **Face + eyes**: MediaPipe Face Mesh gives 468 3D landmarks (nose, eyes, face outline).
- **Gaze / attention**: Nose position relative to face center is used as a simple “head pose” proxy: when you look at the camera, nose is near center; when you turn your head, nose offset grows.
- **Trigger**: After `N` consecutive frames with “looking away”, the app shows the reveal image; when you look back, it hides.

---

## Quick start

**Python:** Use **3.10, 3.11, or 3.12** (MediaPipe does not yet support 3.13+). If your system only has 3.14, use [pyenv](https://github.com/pyenv/pyenv) or another Python 3.12 install and create the venv with that.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run

```bash
python main.py
```

- **ATTENTION** (green) = face detected, nose near center → “looking at screen”.
- **LOOKING AWAY** (red) = nose offset above threshold → “looking away”.
- After about 30 frames of looking away, a second window opens with the **reveal image**. Look back at the camera to hide it.
- Press **`q`** to quit.

### 3. Optional: use your own image

- Put your image at **`assets/reveal.png`** (or any path).
- Or run:  
  `python main.py --image path/to/your_image.png`

### 4. Tune behavior

- **`--frames N`**  
  Number of consecutive “look away” frames before showing the image (default: 30).  
  Example: `python main.py --frames 15`
- **`--camera 1`**  
  Use a different webcam if you have several.

---

## Step-by-step plan (learning path)

1. **Run the starter**  
   Get the pipeline working with the default placeholder image.
2. **Adjust sensitivity**  
   In `gaze_detector.py`, change `look_away_threshold` (e.g. `0.2` = more sensitive, `0.35` = less).
3. **Replace the action**  
   In `main.py`, instead of showing an image you could: blur the screen, log “distracted” time, or call an API.
4. **Improve gaze**  
   Add head pose (yaw/pitch) from more landmarks, or use iris/pupil position for finer gaze direction.
5. **Portfolio upgrade**  
   Add attention score over time, optional AI-generated image when gaze is lost, or a simple dashboard.

---

## Project layout

```
gaze-attention-detector/
├── main.py           # Webcam loop, trigger logic, show reveal image
├── gaze_detector.py  # MediaPipe Face Mesh + “looking at screen” heuristic
├── requirements.txt
├── assets/
│   └── reveal.png    # Optional: image to show when looking away
└── README.md
```

---

## Tech stack

| Layer        | Choice              |
|-------------|----------------------|
| Capture     | OpenCV `VideoCapture` |
| Face/landmarks | MediaPipe Face Mesh |
| Gaze heuristic | Nose offset from face center |
| UI          | OpenCV windows       |

No GPU required; runs on CPU. For a **browser/portfolio** version, you could reimplement the same idea with **MediaPipe JS** or **TensorFlow.js** in a React app.

---

## License

Use and adapt as you like for learning and portfolio projects.
