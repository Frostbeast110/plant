# 🌱 PlantDoc AI - Plant Disease Detection System

A full-stack AI application that detects plant diseases using **image classification** and **text description analysis**. Built with FastAPI backend and React frontend.

---

## ✨ Features

- **🖼️ Image-Based Diagnosis**: Upload leaf/plant images and get instant disease detection
- **💬 Text-Based Diagnosis**: Describe plant symptoms and get disease predictions
- **🤖 Dual AI Models**:
  - **MobileNetV2** (38 plant disease classes, 95.41% accuracy) - Image classification
  - **Transformer Model** - Symptom text classification
- **💊 Treatment Recommendations**: Automatic treatment and management suggestions for detected diseases
- **🌍 Support for 14+ Crop Types**: Apple, Corn, Potato, Tomato, Grape, Pepper, Peach, Strawberry, Blueberry, Cherry, Orange, Raspberry, Soybean, Squash
- **⚡ Fast API Responses**: Real-time predictions with confidence scores
- **🎨 Modern UI**: Beautiful, responsive React frontend with Tailwind CSS

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.8+
- **Node.js**: 14+ (for frontend)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 3GB free space (for models)
- **GPU**: Optional (CUDA-enabled GPU for faster inference)

### Recommended Setup
- Python 3.9+
- Node.js 18+
- 8GB+ RAM
- NVIDIA GPU with CUDA support

---

## 🚀 Installation & Setup

### Prerequisites
Ensure you have installed:
- [Python 3.8+](https://www.python.org/downloads/)
- [Node.js 14+](https://nodejs.org/)
- [Git](https://git-scm.com/)

---

## 📦 Backend Setup (FastAPI)

### Step 1: Navigate to Backend Directory
```powershell
cd bn/backend
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

**Note**: First installation may take 5-10 minutes due to PyTorch and Transformers libraries.

### Step 5: Download Pre-trained Models

The backend includes two pre-trained models:

#### Download MobileNetV2 Plant Disease Model (Image Classification)
```powershell
python download_mobilenet_model.py
```

This downloads the **linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification** model:
- Model: MobileNetV2
- Classes: 38 plant diseases
- Accuracy: 95.41%
- Size: ~60MB
- Location: `models/mobilenet_plantdisease/`

**Expected output:**
```
📥 Downloading model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
💾 Saving to: ./models/mobilenet_plantdisease
Downloading config...
Downloading model weights...
✅ Label map saved (38 classes)
✅ Model downloaded successfully!
```

### Step 6: Verify Backend Files

Ensure these files exist in `models/` directory:
```
models/
├── mobilenet_plantdisease/          # Image classification model
│   ├── config.json
│   ├── model.safetensors
│   ├── label_map.json
│   └── preprocess_config.json
├── best_model/                      # Text classification model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── label_map.json
```

### Step 7: Start Backend Server
```powershell
uvicorn main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Will watch for changes in these directories: ['C:\...\backend']
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
✅ Text model loaded successfully!
✅ MobileNetV2 image model loaded successfully!
```

**✅ Backend is ready when you see both model loading messages!**

---

## 🎨 Frontend Setup (React)

### Step 1: Navigate to Frontend Directory
Open a **NEW terminal** and run:
```powershell
cd bn/frontend
```

### Step 2: Install Dependencies
```powershell
npm install
```

**Note**: This installs React, Tailwind CSS, Axios, and icon libraries. May take 3-5 minutes.

### Step 3: Start Development Server
```powershell
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

**✅ Frontend is ready when the browser opens automatically!**

---

## 📊 Project Directory Structure

```
bn/
├── backend/                          # FastAPI Backend
│   ├── main.py                       # Main FastAPI application
│   ├── requirements.txt              # Python dependencies
│   ├── download_mobilenet_model.py   # Model downloader script
│   ├── models/                       # Pre-trained models
│   │   ├── mobilenet_plantdisease/   # MobileNetV2 model (38 classes)
│   │   ├── best_model/               # Text classifier model
│   │   └── label_map.json            # Disease class mappings
│   └── venv/                         # Python virtual environment
│
├── frontend/                         # React Frontend
│   ├── src/
│   │   ├── App.js                    # Main React component
│   │   ├── App.css                   # Styling
│   │   └── index.js                  # Entry point
│   ├── public/
│   │   └── index.html                # HTML template
│   ├── package.json                  # Node dependencies
│   ├── tailwind.config.js            # Tailwind CSS config
│   └── build/                        # Production build
│
├── README.md                         # This file
└── LICENSE                           # License file
```

---

## 🔌 API Endpoints

### Base URL
```
http://127.0.0.1:8000
```

### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "message": "🌱 Plant Disease Detection API is running successfully!"
}
```

### 2. Image-Based Prediction
```http
POST /predict-image
Content-Type: multipart/form-data
```

**Request:**
- Upload a `.jpg`, `.png`, or `.webp` leaf/plant image

**Response:**
```json
{
  "class_id": 29,
  "label": "Tomato with Early Blight",
  "confidence": 0.9876,
  "recommendation": "Prune lower leaves, avoid water splash, and apply copper-based fungicide."
}
```

### 3. Text-Based Prediction
```http
POST /predict-text
Content-Type: application/json
```

**Request:**
```json
{
  "text": "My tomato plant has yellow spots on the leaves and the stems are brittle."
}
```

**Response:**
```json
{
  "class_id": 28,
  "label": "Tomato with Bacterial Spot",
  "confidence": 0.8734,
  "recommendation": "Use disease-free seeds, apply copper-based bactericides, and avoid overhead watering."
}
```

---

## 💡 Usage Examples

### Testing with cURL

#### Image Prediction
```powershell
$filePath = "C:\path\to\leaf.jpg"
$fileContent = [System.IO.File]::ReadAllBytes($filePath)
$form = @{
    file = $fileContent
}
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict-image" -Method Post -Form $form
```

#### Text Prediction
```powershell
$body = @{
    text = "Brown spots on apple leaves with a velvety texture"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict-text" -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Frontend Usage

1. **For Image Diagnosis**:
   - Click on the image upload area
   - Select a plant/leaf image (JPG, PNG)
   - Click "Get Image Diagnosis"
   - View results with disease label and treatment recommendation

2. **For Text Diagnosis**:
   - Enter symptoms in the text box
   - Example: "Yellow spots, wilting leaves, brittle stems"
   - Click "Get Text Diagnosis"
   - View predicted disease and recommendations

---

## 🤖 Supported Plant Diseases (38 Classes)

### Apple (4 classes)
- Apple Scab
- Apple with Black Rot
- Cedar Apple Rust
- Healthy Apple

### Tomato (11 classes)
- Tomato with Bacterial Spot
- Tomato with Early Blight
- Tomato with Late Blight
- Tomato with Leaf Mold
- Tomato with Septoria Leaf Spot
- Tomato with Spider Mites
- Tomato with Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy Tomato
- + More

### Potato (3 classes)
- Potato with Early Blight
- Potato with Late Blight
- Healthy Potato

### Corn (4 classes)
- Corn with Cercospora Leaf Spot
- Corn with Common Rust
- Corn with Northern Leaf Blight
- Healthy Corn

### Pepper (2 classes)
- Bell Pepper with Bacterial Spot
- Healthy Bell Pepper

### And More...
Grape, Strawberry, Blueberry, Cherry, Peach, Orange, Raspberry, Soybean, Squash

---

## ⚙️ Configuration

### Backend Configuration (main.py)

```python
API_BASE_URL = "http://127.0.0.1:8000"  # Change for production
```

### Frontend Configuration (src/App.js)

```javascript
const API_BASE_URL = 'http://127.0.0.1:8000';  // Backend API URL
```

### CORS Settings

To allow requests from different origins, modify `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🐛 Troubleshooting

### Model Download Issues

**Error**: `Repository Not Found` or `401 Unauthorized`

**Solution**:
```powershell
# Upgrade transformers library
pip install --upgrade transformers torch

# Clear cache
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface\hub

# Try download again
python download_mobilenet_model.py
```

### Backend Won't Start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```powershell
# Ensure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Won't Connect to Backend

**Error**: `Could not connect to the image model API`

**Solution**:
1. Ensure backend is running: `uvicorn main:app --reload --port 8000`
2. Check `API_BASE_URL` in `src/App.js` matches backend URL
3. Clear browser cache (Ctrl + Shift + Delete)
4. Check CORS settings in `main.py`

### Car Photo Shows 99% Corn Disease

**Explanation**: The model only knows plant diseases, so it classifies any image into one of its 38 disease classes. Upload actual plant/leaf images for accurate results.

### Slow Predictions

**Solution**:
- Use GPU if available (NVIDIA GPU recommended)
- Reduce image resolution before upload
- Install CUDA: https://developer.nvidia.com/cuda-downloads

---

## 📈 Performance

| Component | Speed | Hardware |
|-----------|-------|----------|
| Image Prediction | 0.5-2 seconds | CPU (Intel i5) |
| Text Prediction | 0.1-0.5 seconds | CPU |
| Image Prediction | 0.1-0.3 seconds | GPU (NVIDIA) |

---

## 🔐 Security Notes

⚠️ **Current setup is for development only**

For production deployment:
1. Set `allow_origins` to specific frontend domain in CORS
2. Add authentication/authorization
3. Use HTTPS
4. Add rate limiting
5. Validate file uploads (file size, type, content)
6. Deploy behind reverse proxy (Nginx)
7. Use environment variables for sensitive configs

---

## 📝 API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Successful prediction |
| 400 | Bad request (missing file/text) |
| 500 | Server error (model loading issue) |

---

## 🎯 Next Steps

1. **Test the application** with sample plant images
2. **Fine-tune the text model** with domain-specific symptoms
3. **Add user authentication** for production
4. **Deploy to cloud** (AWS, Azure, GCP)
5. **Add mobile app** version

---

## 📚 Model Information

### Image Classification Model (MobileNetV2)
- **Source**: Hugging Face (linkanjarad)
- **Architecture**: MobileNetV2 (lightweight, efficient)
- **Dataset**: Plant Village Dataset (38 classes)
- **Accuracy**: 95.41%
- **Input Size**: 224×224 pixels
- **Framework**: PyTorch via Transformers

### Text Classification Model
- **Type**: DistilBERT-based transformer
- **Task**: Multi-class classification
- **Input**: Plant symptom descriptions
- **Training Data**: Agricultural disease symptoms

---

## 🤝 Contributing

Feel free to:
- Report bugs and issues
- Suggest improvements
- Add new plant disease classes
- Improve UI/UX

---

## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review API endpoint examples
3. Verify model files are downloaded
4. Check terminal output for error messages

---

## 🌟 Credits

- **MobileNetV2 Model**: [linkanjarad](https://huggingface.co/linkanjarad)
- **Dataset**: [Plant Village](https://plantvillage.psu.edu/)
- **Framework**: FastAPI, React, PyTorch, Transformers
- **Icons**: [Lucide React](https://lucide.dev/)

---

**Happy plant disease detection! 🌾🌱**

Last Updated: March 2026
