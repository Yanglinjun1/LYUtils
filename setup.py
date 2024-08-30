import setuptools

setuptools.setup(
    name="ly_utils",
    version="0.1",
    description="A collection of utilities for personal research projects.",
    url="https://github.com/Yanglinjun1/LYUtils",
    author="Linjun Yang",
    install_requires=[
        "pandas",
        "numpy",
        "pingouin==0.5.4",
        "lightning>=2.1.0",
        "torchmetrics>=1.4",
        "torch>=2.0",
        "monai>=1.0.0",
        "timm",
        "segmentation-models-pytorch",
        "pynrrd",
        "slicerio==1.0.0",
        "wandb",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author_email="wadeyang2@gmail.com",
    packages=setuptools.find_packages(),
    zip_safe=False,
    python_requires=">=3.6",
)