from setuptools import setup

setup(
    name="ChEBI-learn",
    version="0.0.0",
    packages=["chem", "chem.models"],
    url="",
    license="",
    author="MGlauer",
    author_email="martin.glauer@ovgu.de",
    description="",
    extras_require={"dev": ["black", "isort", "pre-commit"]},
)
