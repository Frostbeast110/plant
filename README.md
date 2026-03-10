# 🌱 Plant Disease Detection System

A comprehensive AI-powered plant disease detection system that uses both text descriptions and image analysis to identify plant diseases and provide treatment recommendations.

## 📋 Overview

This project consists of a full-stack application with:
- **Backend**: FastAPI server with machine learning models for disease detection
- **Frontend**: Modern React application with intuitive user interface
- **ML Models**: Text-based and image-based classification using transformers and MobileNet

## 🚀 Features

- 🔍 **Dual Detection Methods**: Text description and image upload analysis
- 🧠 **AI-Powered**: Uses state-of-the-art machine learning models
- 💊 **Treatment Recommendations**: Provides specific treatment advice for detected diseases
- 🎨 **Modern UI**: Clean, responsive interface built with React and Tailwind CSS
- ⚡ **Fast API**: High-performance FastAPI backend for quick responses
- 📱 **Cross-Platform**: Works on desktop and mobile devices

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face transformers for NLP
- **TorchVision** - Computer vision library
- **PIL** - Image processing

### Frontend
- **React** - JavaScript library for UI
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client for API calls
- **Lucide React** - Beautiful icons

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models (if not present):**
   The models should be in the `models/` directory. If missing, download them from the appropriate sources.

5. **Run the backend server:**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

## 🎯 Usage

1. **Start both servers** (backend and frontend) as described above
2. **Open your browser** and go to `http://localhost:3000`
3. **Choose detection method:**
   - **Text Analysis**: Describe the plant symptoms in text
   - **Image Analysis**: Upload a photo of the affected plant
4. **Get Results**: View disease prediction and treatment recommendations

## 📁 Project Structure

```
plant-disease-detection/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── models/                 # ML models directory
│   │   ├── best_model/         # Text classification model
│   │   └── mobilenet_plantdisease/  # Image classification model
│   └── recommendations.json    # Treatment recommendations
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   └── index.css          # Styles
│   ├── package.json           # Node dependencies
│   └── public/                # Static files
└── README.md                  # This file
```

## 🔧 API Endpoints

### POST `/predict/text`
Predict disease from text description.

**Request Body:**
```json
{
  "text": "Brown spots on tomato leaves"
}
```

### POST `/predict/image`
Predict disease from image upload.

**Request Body:** Form data with `file` field containing image.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Plant disease dataset providers
- Hugging Face for transformers library
- FastAPI and React communities
- Open source contributors

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Happy Gardening! 🌿**