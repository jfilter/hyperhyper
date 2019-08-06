from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

classifiers = [
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "License :: OSI Approved :: MIT License",
    "Topic :: Utilities",
]

setup(
    name="hyperhyper",
    version="0.0.0",
    python_requires=">=3.6",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jfilter/hyper-hyper",
    author="Johannes Filter",
    author_email="hi@jfilter.de",
    license="MIT",
    packages=["hyperhyper"],
    classifiers=classifiers,
    install_requires=[
        "dataset==1.*",
        "tqdm",
        "gensim==3.*",
        "importlib_resources ; python_version<'3.7'",
    ],
    extras_require={"full": ["spacy==2.*", "scikit-learn"]},
)

