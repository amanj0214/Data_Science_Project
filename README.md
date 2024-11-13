# Data Science Project

A Sample Data Science Project for understanding the structure


### Generating `requirements.txt`
If you are working in a virtual environment and have installed packages using pip, you can easily generate a requirements.txt file that includes all installed packages with their versions. Here’s how to do it:
1. Create `venv`
```
python -m venv venv
```
2. Activate your virtual environment: If you are using a virtual environment (recommended), activate it first. For example:
```
# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```
3. Generate the requirements.txt file: Use the following command:
```
python generate_requirements.py
```

pip install -r requirements.txt