# Blackspot-Detection-Alert-System

### Overview
An ML-powered geospatial analysis tool that identifies accident-prone 'black spots' using clustering and provides real-time alerts for travelers.

### Tech Stack
- Python (ML Engine)
- React Native (Expo)
- FastAPI

### Getting Started - Quick Setup
```bash
# Clone repo
git clone https://github.com/road-safety-and-blackspot-detection/Blackspot-Detection-Alert-System.git
cd Blackspot-Detection-Alert-System

# Backend (ML + FastAPI)
cd ml-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload

# Frontend (React Native Expo)
cd ../mobile-app
npm install
npx expo start
```
For detailed setup & workflow:  [docs/instructions.md](docs/instructions.md)