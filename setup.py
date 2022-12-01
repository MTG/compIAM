from setuptools import find_packages, setup

with open("./requirements.txt") as f:
    REQUIREMENTS = f.readlines()

setup(
    name="compiam",
    version="0.1.0",
    author_email=["genis.plaja@upf.edu", "thomas.nuttall@upf.edu"],
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(exclude=["test", "*.test", "*.test.*"]),
    package_data={
        "compiam": [
            "models/*",
            "conf/*",
            "visualisation/waveform_player/waveform-playlist/*",
            "utils/augmentation/*",
            "utils/NMFtoolbox/*",
            "notebooks/*"
        ]
    },
    long_description=open("./README.md").read(),
    install_requires=REQUIREMENTS,
    extras_require={
        "tests": [
            "pytest>=4.4.0",
            "pytest-cov>=2.6.1",
        ],
        "docs": [
            "numpydoc",
            "recommonmark",
            "sphinx>=3.4.0",
            "sphinxcontrib-napoleon",
            "sphinx_rtd_theme",
        ],
    },
)
