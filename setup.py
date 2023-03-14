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
        "torch>=1.12",
        "matplotlib",
        "numpy",
        "mmcv>=1.4.7",
        "pytorch-lightning==1.7.7",
        "pytest",
        "setuptools>=59.5.0",
        "wandb",
        "plotly",
        "scipy",
        "gradio",
        "einops",
        "pydantic",
    ],
)
