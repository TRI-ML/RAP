from setuptools import setup

setup(
    name="risk_biased",
    version="0.1",
    description="Risk biased trajectory prediction",
    authors=["Haruki Nishimura", "Jean Mercat"],
    author_emails=["haruki.nishimura@tri.global", "jean.mercat@tri.global"],
    license="MIT",
    packages=["risk_biased"],
    zip_safe=False,
    install_requires=[
        "torch>=2.1.0",
        "matplotlib",
        "numpy==1.26.4", 
        "mmcv-lite",  # Use mmcv-lite instead of mmcv for compatibility
        "lightning>=2.1.0",  # Using lightning instead of pytorch-lightning
        "pytest",
        "setuptools>=59.5.0",
        "wandb",
        "plotly",
        "scipy",
        "pydantic>=2.0.0",
        "gradio",
        "einops",
        "torchmetrics>=1.0.0"
    ],
)
