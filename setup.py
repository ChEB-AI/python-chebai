from setuptools import find_packages, setup

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
    # `|` operator used for type hint in chebai/loss/semantic.py is only supported for python_version >= 3.10.
    # https://stackoverflow.com/questions/76712720/typeerror-unsupported-operand-types-for-type-and-nonetype
    # python_requires="<=3.11.8",
    python_requires=">=3.10.0, <3.11.8",
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
        "omegaconf",
        "seaborn",
        "deepsmiles",
        "iterative-stratification",
        "wandb",
        "chardet",
        # --- commented below due to strange dependency error while setting up new env
        # "yaml",`
        "torchmetrics",
    ],
    extras_require={"dev": ["black", "isort", "pre-commit"]},
)
