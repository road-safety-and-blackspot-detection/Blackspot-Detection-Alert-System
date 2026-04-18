# Setup and Workflow

## Requirements
Python 3.12 recommended

How to work on this:

Ooen a folder in vscode where you keep your all projects
open a new terminal: ctrl + `

```bash
git clone https://github.com/road-safety-and-blackspot-detection/Blackspot-Detection-Alert-System.git
cd Blackspot-Detection-Alert-System

# Python Setup - Install venv for python
cd ml-engine
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt

# Setup react-native expo
cd mobile-app
npm install
npx expo start
# press "c" for QR, scan it with ExpoGo app
```

## Git First Time Setup

```bash
# Always start from main
git checkout main
git pull origin main

# Create your own branch
git checkout -b feature/your-own-branch-name

# Work normally and Save your work
git add .
git commit -m "Add small description"
git push origin feature/your-own-branch-name
```


## Daily Workflow before coding anything
```bash
git checkout main
git pull origin main
git checkout your-branch
```
## Daily Workflow after editing files
```bash
git add .
git commit -m "Add small description"
git push origin feature/my-own-branch-name
```
