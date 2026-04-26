# Setup and Workflow

## Requirements
Python 3.12 recommended

How to work on this:

Ooen a folder in vscode where you keep your all projects
open a new terminal: ctrl + ~

```bash
git clone https://github.com/road-safety-and-blackspot-detection/Blackspot-Detection-Alert-System.git
cd Blackspot-Detection-Alert-System

# Python Setup - Install venv for python
cd ml-engine
python -m venv venv
venv\Scripts\activate #(For Windows users)
source venv/bin/activate #(For mac/linux users)
pip install -r requirements.txt

# Setup react-native expo
cd mobile-app
npm install
npx expo start
# press "c" for QR, scan it with ExpoGo app
```

## Git First Time Setup

```bash
# Always start from main (switch to main branch)
git switch main 
git pull origin main

# Create your own branch and switch to it
git switch -c feature/your-own-branch-name

# Work normally and Save your work
git add .
git commit -m "Add small description"
git push -u origin feature/your-own-branch-name
# -u links branch → so next time push is easier
```


## Daily Workflow before coding anything
```bash
# Go to main and get latest changes
git switch main
git pull origin main

# Switch back to your branch
git switch your-own-branch-name

# Update your branch with latest main (IMPORTANT)
git merge main
```
## Daily Workflow after editing files
```bash
git add .
git commit -m "Add small description"
git push 
#At last, go to GitHub and open a Pull Request to merge into main
```
