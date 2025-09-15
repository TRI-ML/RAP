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
        "torch>=1.13.1",
        "matplotlib",
        "numpy==1.26.4",
        "mmcv==1.4.7",
        "pytorch-lightning>=1.8.6,<2.0.0",
        "pytest",
        "setuptools>=59.5.0",
        "wandb",
        "plotly",
        "scipy",
        "pydantic>=1.10,<2.0.0",
        "gradio",
        "einops",
        "torchmetrics>=0.11.4,<1.0.0"
    ],
)
