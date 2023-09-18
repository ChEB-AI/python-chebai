from setuptools import setup
from setuptools import find_packages

packages = find_packages()
print(packages)
setup(
    name="chebai",
    version="0.0.2.dev0",
    packages=packages,
    package_data={"": ["**/*.txt", "**/*.json"]},
    include_package_data=True,
    url="",
    license="",
    author="MGlauer",
    author_email="martin.glauer@ovgu.de",
    description="",
    zip_safe=False,
    install_requires=[
        "certifi",
        "idna",
        "joblib",
        "networkx",
        "numpy",
        "pandas",
        "python-dateutil",
        "pytz",
        "requests",
        "scikit-learn",
        "scipy",
        "six",
        "threadpoolctl",
        "torch",
        "typing-extensions",
        "urllib3",
        "transformers",
        "fastobo",
        "pysmiles",
        "scikit-network",
        "svgutils",
        "matplotlib",
        "rdkit",
        "selfies",
        "lightning",
        "jsonargparse[signatures]>=4.17.0",
        "omegaconf"
    ],
    extras_require={"dev": ["black", "isort", "pre-commit"]},
)
