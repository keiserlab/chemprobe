from setuptools import setup, find_packages

setup(
    name="chemprobe",
    version="0.1.5",
    author="William Connell",
    author_email="connell@keiserlab.org",
    description="A package for chemprobe",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/keiserlab/chemprobe",
    packages=find_packages(),
    install_requires=[
        "pandas==1.5.1",
        "scikit-learn==1.1.3",
        "pytorch-lightning==1.8.4",
        "optuna==2.10.1",
        "rdkit==2022.9.5",
        "thunor==0.1.29",
        "kneed==0.8.2",
        "captum==0.7.0",
        "torchmetrics==0.10.3",
        "gdown==4.6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7,<=3.11.0',
    include_package_data=True,
    package_data={
        "chemprobe": ["data/cpds.csv.gz"]
    },
)
