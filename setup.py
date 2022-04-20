from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="textfier",
    version="1.0.2",
    description="Text-based Modifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gustavo Rosa",
    author_email="gustavo.rosa@unesp.br",
    url="https://github.com/gugarosa/textfier",
    license="Apache 2.0",
    install_requires=[
        "coverage>=5.5",
        "nltk>=3.5",
        "pandas>=1.2.3",
        "pre-commit>=2.17.0",
        "pylint>=2.7.2",
        "pytest>=6.2.2",
        "scikit-learn>=0.24.1",
        "transformers>=4.3.3",
    ],
    extras_require={
        "tests": [
            "coverage",
            "pytest",
            "pytest-pep8",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
