from setuptools import setup, find_packages

setup(
    name="chest-xray-classifier",
    version="1.0.0",
    description="AI-powered Chest X-ray image classification app using FastAPI and PyTorch",
    author="Jay Gupta",
    author_email="ae21b026@smail.iitm.ac.in",
    packages=find_packages(include=["training*", "tuning*"]),
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: FastAPI",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)