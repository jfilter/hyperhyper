from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

classifiers = [
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "License :: OSI Approved :: BSD License",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

setup(
    name="hyperhyper",
    version="0.1.0",
    python_requires=">=3.6",
    description="Python Library to Construct Word Embeddings for Small Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jfilter/hyperhyper",
    author="Johannes Filter",
    author_email="hi@jfilter.de",
    license="BSD",
    packages=find_packages(),
    package_data={
        # include evaluation datasets
        "": ["*.txt"]
    },
    zip_safe=True,
    classifiers=classifiers,
    install_requires=[
        "dataset==1.*",
        "tqdm",
        "gensim==3.*",
        "importlib_resources ; python_version<'3.7'",
    ],
    extras_require={"full": ["spacy==2.*", "scikit-learn"]},
)

