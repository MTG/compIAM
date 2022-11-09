import os
import pathlib
from setuptools import find_packages, setup

WORKDIR = os.path.dirname(pathlib.Path(__file__).parent.resolve())

with open(os.path.join(WORKDIR, "requirements.txt")) as f:
    REQUIREMENTS = f.readlines()

setup(
    name="compiam",
    version="0.1.0",
    packages=find_packages(),
    author_email=["thomas.nuttall@upf.edu", "genis.plaja@upf.edu"],
    zip_safe=False,
    include_package_data=True,
    long_description=open(os.path.join(WORKDIR, "README.md")).read(),
    install_requires=REQUIREMENTS,
    extras_require={
        "tests": [
            "pytest>=4.4.0",
            "pytest-cov>=2.6.1",
        ],
        "docs": [
            "sphinx>=3.4.0",
            "sphinxcontrib-napoleon",
            "sphinx_rtd_theme",
        ],
    },
)
