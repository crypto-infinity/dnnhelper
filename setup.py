from setuptools import setup, find_packages

setup(
    name="dnnhelper",
    version="0.1.0",
    description="Helper module for Deep Neural Networks (DNN) with PyTorch.",
    author="Gabriele Scorpaniti",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/dnnhelper",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchmetrics",
        "scikit-learn",
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn"
    ],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
