from setuptools import setup, find_packages

setup(
    name="drowsiness_detection",
    version="0.1.0",
    description="Real-time driver drowsiness detection AI model",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.5.0",
        "tensorflow>=2.16.0",
        "pytest>=8.0.0"
    ],
    python_requires=">=3.9",
)