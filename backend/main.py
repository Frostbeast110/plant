from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForImageClassification
from PIL import Image
import io
import json
from pathlib import Path
from torchvision import transforms

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
TEXT_MODEL_PATH = BASE_DIR / "models" / "best_model"
IMAGE_MODEL_PATH = BASE_DIR / "models" / "mobilenet_plantdisease"
LABEL_MAP_PATH = BASE_DIR / "models" / "mobilenet_plantdisease" / "label_map.json"
API_BASE_URL = "http://127.0.0.1:8000"  # Update if deploying

# ----------------------------
# FastAPI App Initialization
# ----------------------------
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using text and image inputs"
)

# Enable CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load Label Map
# ----------------------------
try:
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Label map not found at {LABEL_MAP_PATH}.")
except Exception as e:
    raise RuntimeError(f"Failed to load label map: {str(e)}")

# ----------------------------
# 🔹 Disease Recommendation Map (38 Classes)
# ----------------------------
disease_recommendations = {
    "Apple Scab": "Remove infected leaves, apply fungicide like Mancozeb, and ensure good air circulation.",
    "Apple with Black Rot": "Prune infected branches, remove mummified fruits, and spray Captan or copper-based fungicide.",
    "Cedar Apple Rust": "Remove nearby juniper hosts, and use fungicides like Myclobutanil.",
    "Healthy Apple": "No treatment needed. Continue regular care and monitoring.",
    "Healthy Blueberry Plant": "No treatment needed. Continue regular care and monitoring.",
    "Cherry with Powdery Mildew": "Spray with sulfur-based fungicide or neem oil, and improve air circulation.",
    "Healthy Cherry Plant": "No treatment needed. Continue regular care and monitoring.",
    "Corn (Maize) with Cercospora and Gray Leaf Spot": "Rotate crops, use resistant hybrids, and apply strobilurin fungicides.",
    "Corn (Maize) with Common Rust": "Plant resistant varieties and apply fungicides like Azoxystrobin if infection is severe.",
    "Corn (Maize) with Northern Leaf Blight": "Use resistant hybrids and rotate crops to minimize spore survival.",
    "Healthy Corn (Maize) Plant": "No treatment needed. Continue regular care and monitoring.",
    "Grape with Black Rot": "Prune and destroy infected vines, and apply myclobutanil during early growth.",
    "Grape with Esca (Black Measles)": "Prune infected wood and apply copper-based fungicide to cut surfaces.",
    "Grape with Isariopsis Leaf Spot": "Remove affected leaves and apply copper sulfate fungicide.",
    "Healthy Grape Plant": "No treatment needed. Continue regular care and monitoring.",
    "Orange with Citrus Greening": "Control psyllid vector, remove infected trees, and plant disease-free stock.",
    "Peach with Bacterial Spot": "Apply copper-based fungicides before leaf drop and plant resistant varieties.",
    "Healthy Peach Plant": "No treatment needed. Continue regular care and monitoring.",
    "Bell Pepper with Bacterial Spot": "Use certified seeds, apply copper sprays, and avoid working with wet plants.",
    "Healthy Bell Pepper Plant": "No treatment needed. Continue regular care and monitoring.",
    "Potato with Early Blight": "Use certified seed, apply Chlorothalonil or Mancozeb, and avoid overhead irrigation.",
    "Potato with Late Blight": "Use resistant varieties, remove infected plants, and spray metalaxyl-based fungicide.",
    "Healthy Potato Plant": "No treatment needed. Continue regular care and monitoring.",
    "Healthy Raspberry Plant": "No treatment needed. Continue regular care and monitoring.",
    "Healthy Soybean Plant": "No treatment needed. Continue regular care and monitoring.",
    "Squash with Powdery Mildew": "Apply sulfur-based fungicide, improve air circulation, and remove heavily infected leaves.",
    "Strawberry with Leaf Scorch": "Remove infected leaves, avoid overhead watering, and apply fungicides like captan.",
    "Healthy Strawberry Plant": "No treatment needed. Continue regular care and monitoring.",
    "Tomato with Bacterial Spot": "Use disease-free seeds, apply copper-based bactericides, and avoid overhead watering.",
    "Tomato with Early Blight": "Prune lower leaves, avoid water splash, and apply copper-based fungicide.",
    "Tomato with Late Blight": "Destroy infected plants and spray with copper oxychloride or Mancozeb.",
    "Tomato with Leaf Mold": "Increase air circulation, avoid overcrowding, and apply fungicide like chlorothalonil.",
    "Tomato with Septoria Leaf Spot": "Remove diseased leaves, use drip irrigation, and apply fungicide like mancozeb.",
    "Tomato with Spider Mites or Two-spotted Spider Mite": "Spray with insecticidal soap or neem oil, increase humidity, and remove heavily infested leaves.",
    "Tomato with Target Spot": "Avoid water on foliage, use crop rotation, and apply fungicides such as difenoconazole.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies, remove infected plants, and use virus-resistant varieties.",
    "Tomato Mosaic Virus": "Avoid handling wet plants, disinfect tools, and use resistant varieties.",
    "Healthy Tomato Plant": "No treatment needed. Continue regular care and monitoring."
}

# ----------------------------
# Load Text Model
# ----------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(str(TEXT_MODEL_PATH), local_files_only=True)
    text_model = AutoModelForSequenceClassification.from_pretrained(str(TEXT_MODEL_PATH), local_files_only=True)
    text_model.eval()
    print("✅ Text model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading text model from {TEXT_MODEL_PATH}: {str(e)}")
    raise RuntimeError("Failed to load text model. Check local model path and model files.")

# ----------------------------
# Load Image Model (MobileNetV2)
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    print(f"Loading image model from: {IMAGE_MODEL_PATH}")
    print(f"Number of classes: {len(label_map)}")
    
    # Load MobileNetV2 model
    image_model = AutoModelForImageClassification.from_pretrained(str(IMAGE_MODEL_PATH), local_files_only=True)
    image_model.to(device)
    image_model.eval()

    print("✅ MobileNetV2 image model loaded successfully!")

except Exception as e:
    import traceback
    traceback.print_exc()
    raise RuntimeError(f"Failed to load image model: {str(e)}")

# ----------------------------
# Image Preprocessing (MobileNetV2)
# ----------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "🌱 Plant Disease Detection API is running successfully!"}

# ----------------------------
# Text Prediction Endpoint
# ----------------------------
class TextRequest(BaseModel):
    text: str

@app.post("/predict-text")
async def predict_text(request: TextRequest):
    """Predict disease based on text description."""
    try:
        text = request.text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = text_model(**inputs)

        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class_id = torch.argmax(predictions, dim=1).item()
        confidence = float(predictions[0][predicted_class_id])

        disease_label = label_map.get(str(predicted_class_id), f"Unknown Disease (ID: {predicted_class_id})")

        # 🔹 Add recommendation system
        recommendation = disease_recommendations.get(
            disease_label,
            "No specific treatment found. Consult an agricultural expert."
        )

        return {
            "class_id": predicted_class_id,
            "label": disease_label,
            "confidence": round(confidence, 4),
            "recommendation": recommendation
        }

    except Exception as e:
        print(f"Text prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Text prediction failed due to internal error.")

# ----------------------------
# Image Prediction Endpoint
# ----------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Predict disease based on uploaded leaf image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        image_tensor = image_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = image_model(image_tensor)

        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class_id = torch.argmax(predictions, dim=1).item()
        confidence = float(predictions[0][predicted_class_id])

        disease_label = label_map.get(str(predicted_class_id), f"Unknown Disease (ID: {predicted_class_id})")

        # 🔹 Add recommendation system
        recommendation = disease_recommendations.get(
            disease_label,
            "No specific treatment found. Consult an agricultural expert."
        )

        return {
            "class_id": predicted_class_id,
            "label": disease_label,
            "confidence": round(confidence, 4),
            "recommendation": recommendation
        }

    except Exception as e:
        print(f"Image prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image prediction failed due to internal error.")

# ----------------------------
# Run the App
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
