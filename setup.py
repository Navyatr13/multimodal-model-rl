from setuptools import setup, find_packages

setup(
    name="multimodal_model_rl",
    version="0.1.0",
    author="Your Name",
    description="Multimodal Model with Reinforcement Learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "gym",
        "numpy",
        "pandas",
        "matplotlib",
        "torchtext",
        "torch-geometric",
        "Pillow",
    ],
    python_requires=">=3.7",
)