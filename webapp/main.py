from fastapi import FastAPI, File, UploadFile
import gdown
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, regularizers
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Rebuild model architecture & load weights ──
DRIVE_FILE_ID  = "1K98p7XcEXNaeXHUp-k_lBIlqfVMBsfsi"

# Check local path first, then download if not found
LOCAL_WEIGHTS  = r"D:\6th Sem\Project - II\saved_models\plant_disease_weights.weights.h5"
WEIGHTS_PATH   = "plant_disease_weights.weights.h5"

if os.path.exists(LOCAL_WEIGHTS):
    WEIGHTS_PATH = LOCAL_WEIGHTS
    print("✅ Using local weights!")
elif not os.path.exists(WEIGHTS_PATH):
    print("Downloading model weights from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
                   WEIGHTS_PATH, quiet=False)
    print("✅ Weights downloaded!")

CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.txt")
NUM_CLASSES      = 15
IMG_SHAPE        = (224, 224, 3)

print("Building model...")
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

inputs  = tf.keras.Input(shape=IMG_SHAPE)
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = tf.keras.Model(inputs, outputs)

model.load_weights(WEIGHTS_PATH)
print("✅ Model ready!")

# ── Load class names ──
class_map = {}
with open(CLASS_NAMES_PATH, 'r') as f:
    for line in f:
        idx, cls = line.strip().split(': ', 1)
        class_map[int(idx)] = cls

# ── Disease info database ──
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "cause"      : "Caused by Xanthomonas bacteria, spreads in warm humid conditions.",
        "symptoms"   : "Small water-soaked spots on leaves that turn brown with yellow halo.",
        "solution"   : "Apply copper-based fungicides. Remove infected leaves. Avoid overhead watering.",
        "prevention" : "Rotate crops, use resistant varieties, maintain proper spacing."
    },
    "Pepper__bell___healthy": {
        "cause"      : "No disease detected.",
        "symptoms"   : "Plant appears healthy.",
        "solution"   : "Continue regular care — proper watering, fertilization and sunlight.",
        "prevention" : "Maintain good agricultural practices."
    },
    "Potato___Early_blight": {
        "cause"      : "Caused by Alternaria solani fungus. Favored by warm, humid weather.",
        "symptoms"   : "Dark brown spots with concentric rings on older leaves.",
        "solution"   : "Apply Mancozeb or Chlorothalonil fungicides. Remove infected leaves.",
        "prevention" : "Crop rotation, destroy infected debris, use certified disease-free seeds."
    },
    "Potato___Late_blight": {
        "cause"      : "Caused by Phytophthora infestans. Spreads in cool, wet conditions.",
        "symptoms"   : "Water-soaked lesions on leaves turning brown-black. White mold underneath.",
        "solution"   : "Apply Metalaxyl or Cymoxanil fungicides. Destroy infected plants immediately.",
        "prevention" : "Plant resistant varieties, avoid excessive irrigation."
    },
    "Potato___healthy": {
        "cause"      : "No disease detected.",
        "symptoms"   : "Plant appears healthy.",
        "solution"   : "Continue regular care — proper watering, fertilization and sunlight.",
        "prevention" : "Maintain good agricultural practices."
    },
    "Tomato_Bacterial_spot": {
        "cause"      : "Caused by Xanthomonas bacteria. Spreads through infected seeds and rain.",
        "symptoms"   : "Small dark spots with yellow halos on leaves and fruit.",
        "solution"   : "Apply copper-based bactericides. Remove infected plant parts.",
        "prevention" : "Use disease-free seeds, crop rotation, avoid overhead irrigation."
    },
    "Tomato_Early_blight": {
        "cause"      : "Caused by Alternaria solani fungus.",
        "symptoms"   : "Brown spots with concentric rings on older leaves.",
        "solution"   : "Apply Mancozeb fungicide. Remove lower infected leaves. Mulch around plants.",
        "prevention" : "Crop rotation, proper plant spacing, avoid wetting foliage."
    },
    "Tomato_Late_blight": {
        "cause"      : "Caused by Phytophthora infestans. Thrives in cool, moist conditions.",
        "symptoms"   : "Large irregular brown patches. White mold on undersides in humid conditions.",
        "solution"   : "Apply systemic fungicides immediately. Remove and destroy infected plants.",
        "prevention" : "Plant resistant varieties, improve air circulation, avoid evening watering."
    },
    "Tomato_Leaf_Mold": {
        "cause"      : "Caused by Passalora fulva fungus. Favored by high humidity.",
        "symptoms"   : "Yellow spots on upper leaf, olive-green mold on underside.",
        "solution"   : "Apply Chlorothalonil. Improve ventilation. Reduce humidity.",
        "prevention" : "Use resistant varieties, ensure good air circulation."
    },
    "Tomato_Septoria_leaf_spot": {
        "cause"      : "Caused by Septoria lycopersici fungus.",
        "symptoms"   : "Small circular spots with dark borders and lighter centers.",
        "solution"   : "Apply Mancozeb or copper fungicides. Remove infected lower leaves.",
        "prevention" : "Crop rotation, mulching, avoid overhead watering."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "cause"      : "Infestation by Tetranychus urticae mites. Worse in hot, dry conditions.",
        "symptoms"   : "Tiny yellow speckles on leaves, fine webbing on undersides.",
        "solution"   : "Apply miticides or neem oil. Spray water on undersides of leaves.",
        "prevention" : "Maintain adequate humidity, monitor regularly."
    },
    "Tomato__Target_Spot": {
        "cause"      : "Caused by Corynespora cassiicola fungus.",
        "symptoms"   : "Brown spots with concentric rings on leaves and fruit.",
        "solution"   : "Apply Azoxystrobin or Chlorothalonil. Remove infected leaves.",
        "prevention" : "Crop rotation, proper plant spacing, avoid overhead irrigation."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "cause"      : "Viral disease transmitted by whiteflies.",
        "symptoms"   : "Yellowing and upward curling of leaves, stunted growth.",
        "solution"   : "No cure. Remove infected plants. Control whitefly with insecticides.",
        "prevention" : "Use virus-resistant varieties, control whiteflies, use insect nets."
    },
    "Tomato__Tomato_mosaic_virus": {
        "cause"      : "Caused by Tomato Mosaic Virus, spreads through contact and tools.",
        "symptoms"   : "Mosaic pattern of light and dark green on leaves, leaf distortion.",
        "solution"   : "No chemical cure. Remove infected plants. Disinfect tools with bleach.",
        "prevention" : "Use virus-free seeds, wash hands, control aphids."
    },
    "Tomato_healthy": {
        "cause"      : "No disease detected.",
        "symptoms"   : "Plant appears healthy.",
        "solution"   : "Continue regular care — proper watering, fertilization and sunlight.",
        "prevention" : "Maintain good agricultural practices."
    }
}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top3  = preds.argsort()[-3:][::-1]

    results = []
    for idx in top3:
        cls  = class_map[idx]
        conf = float(preds[idx] * 100)
        info = DISEASE_INFO.get(cls, {})
        results.append({
            "class"      : cls,
            "confidence" : round(conf, 2),
            "cause"      : info.get("cause", ""),
            "symptoms"   : info.get("symptoms", ""),
            "solution"   : info.get("solution", ""),
            "prevention" : info.get("prevention", "")
        })

    return JSONResponse({"predictions": results})

@app.post("/chat")
async def chat(data: dict):
    question = data.get("message", "").lower()
    disease  = data.get("disease", "")
    info     = DISEASE_INFO.get(disease, {})

    if any(w in question for w in ["cause", "why", "reason"]):
        reply = f"🔬 Cause: {info.get('cause', 'No info available.')}"
    elif any(w in question for w in ["symptom", "sign", "look"]):
        reply = f"🔍 Symptoms: {info.get('symptoms', 'No info available.')}"
    elif any(w in question for w in ["treat", "cure", "fix", "solution", "medicine"]):
        reply = f"💊 Solution: {info.get('solution', 'No info available.')}"
    elif any(w in question for w in ["prevent", "avoid", "stop"]):
        reply = f"🛡️ Prevention: {info.get('prevention', 'No info available.')}"
    elif any(w in question for w in ["hello", "hi", "hey"]):
        reply = "👋 Hello! I'm your Plant Disease Assistant. Upload a leaf image and ask me anything!"
    else:
        reply = ("I can help you with:\n"
                 "• Cause — ask 'what is the cause?'\n"
                 "• Symptoms — ask 'what are the symptoms?'\n"
                 "• Solution — ask 'how to treat it?'\n"
                 "• Prevention — ask 'how to prevent it?'")

    return JSONResponse({"reply": reply})

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
