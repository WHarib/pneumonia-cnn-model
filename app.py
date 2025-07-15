from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import zipfile, os, io, cv2, traceback

app = FastAPI(
    title="Pneumonia CNN Classifier API",
    description="TensorFlow CNN (~90 % accuracy) for pneumonia detection",
    docs_url="/docs",
)

# ───── Paths ──────────────────────────────────────────────────────────────
ZIP_FILE   = "pneumonia_cnn_saved_model.zip"
MODEL_DIR  = "/tmp/pneumonia_cnn_saved_model"   # <─ always writable
CLASS_NAMES = ["PNEUMONIA", "NORMAL"]

# ───── Un-zip SavedModel (first launch) ───────────────────────────────────
if not os.path.exists(MODEL_DIR):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_FILE, "r") as zf:
            zf.extractall(MODEL_DIR)
    except Exception as e:
        raise RuntimeError(f"Cannot extract SavedModel: {e}")

# ───── Load model ─────────────────────────────────────────────────────────
try:
    model  = tf.saved_model.load(MODEL_DIR)
    infer  = model.signatures["serving_default"]
    IN_KEY  = list(infer.structured_input_signature[1].keys())[0]
    OUT_KEY = list(infer.structured_outputs.keys())[0]
except Exception as e:
    raise RuntimeError(f"Model loading error: {e}\n{traceback.format_exc()}")

# ───── Helpers ────────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """RGB PIL → (1,150,150,1) float32 0-1"""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (150, 150)) / 255.0
    return gray.reshape(1, 150, 150, 1).astype("float32")

# ───── Routes ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img   = Image.open(io.BytesIO(await file.read())).convert("RGB")
        batch = preprocess(img)

        outputs = infer(**{IN_KEY: tf.convert_to_tensor(batch)})
        prob    = float(tf.sigmoid(outputs[OUT_KEY])[0][0])
        label   = CLASS_NAMES[0] if prob > 0.5 else CLASS_NAMES[1]
        conf    = prob if label == "PNEUMONIA" else 1 - prob

        return JSONResponse(
            {"diagnosis": label, "confidence": round(conf, 4), "raw_score": prob}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
