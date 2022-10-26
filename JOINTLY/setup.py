import sys
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError("JOINTLY requires Python 3")

# get version
#with open("src/JOINTLY/version.py") as f:
#    exec(f.read())
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="JOINTLY",
    version='0.0.1',
    description="Jointly for joint clustering of scRNA-seq datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/madsen-lab/JOINTLY",
    author=['Andreas Fønss Møller','Jesper Grud Skat Madsen'],
    author_email=['andreasfm@bmb.sdu.dk', 'jgsm@imada.sdu.dk'],
    package_dir={"": "src"},
    packages=["jointly"],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scanpy",
        "graphtools",
        "ray",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
