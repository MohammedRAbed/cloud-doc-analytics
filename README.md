# 📁 Document Analytics Cloud System

![Project Banner](https://via.placeholder.com/1200x400?text=Document+Analytics+Cloud+Banner) <!-- Replace with actual banner image -->

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![Firebase](https://img.shields.io/badge/Firebase-9.0+-orange.svg)](https://firebase.google.com)

A cloud-based document processing system with:
- 📤 Document upload/storage
- 🔍 Advanced search with highlighting
- 🗂 ML-powered classification
- 📊 Analytics dashboard

## 🖥️ Screenshots

| Feature | Preview |
|---------|---------|
| **Dashboard** | ![Dashboard](https://via.placeholder.com/600x300?text=Dashboard+Screenshot) |
| **Document Upload** | ![Upload](https://via.placeholder.com/600x300?text=Upload+Interface) |
| **Search Results** | ![Search](https://via.placeholder.com/600x300?text=Search+Results+with+Highlighting) |
| **Classification** | ![Classification](https://via.placeholder.com/600x300?text=Classification+Results) |

## 🚀 Features

- **Multi-format Support**: PDF, DOCX, and DOC files
- **Smart Processing**:
  - Title extraction from content
  - Word count analysis
- **Cloud-Native Architecture**:
  - Firebase Storage for documents
  - Firestore for metadata
- **Responsive UI**: Works on desktop and mobile

## 🛠️ Tech Stack

**Frontend**:
- Bootstrap 5
- Chart.js
- Firebase SDK

**Backend**:
- Python FastAPI
- PyMuPDF (PDF processing)
- python-docx (Word processing)
- scikit-learn (Classification)

**Cloud Services**:
- Firebase Hosting
- Cloud Functions
- Firestore Database
- Cloud Storage

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- Node.js 16+
- Firebase CLI (`npm install -g firebase-tools`)

### Installation
```bash
git clone https://github.com/yourusername/document-analytics-cloud.git
cd document-analytics-cloud

# Backend setup
cd backend
pip install -r requirements.txt

# Firebase setup
firebase init emulators
```

### Running Locally
```bash
# Start backend
uvicorn main:app --reload

# Start frontend with emulators
firebase emulators:start
```

Access the system at: http://localhost:8000

## 📂 Project Structure
```
document-analytics-cloud/
├── frontend/           # Web interface
│   ├── public/         # Static assets
│   └── src/            # JS/CSS files
├── backend/            # FastAPI application
│   ├── models/         # Data models
│   └── services/       # Business logic
├── firebase/           # Firebase config
└── docs/               # Documentation
```

## 📚 Documentation
- [API Reference](docs/API.md)
- [Architecture Diagram](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)

## ⚠️ Limitations
- Requires Firebase billing account for full deployment
- Classification accuracy depends on training data
- Web scraping feature is experimental

## 📜 License
MIT License - See [LICENSE](LICENSE) for details

---

**Developed by**: [Mohammed Abed]  
**Course**: Cloud and Distributed Systems (SICT 4313)  
**Institution**: Islamic University of Gaza  
**Instructor**: Dr. Rebhi S. Baraka
