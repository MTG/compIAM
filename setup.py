from setuptools import find_packages, setup

setup(
    name='compiam',
    version="1.0",
    packages=find_packages(),
    author_email=['thomas.nuttall@upf.edu','genis.plaja@upf.edu'],
    zip_safe=False,
    include_package_data=True,
    long_description=open('README.md').read(),
    install_requires = [
        "matplotlib>=3.0.0",
        "numpy==1.18.5",
        "librosa==0.8.0",
        "SoundFile==0.10.3.post1",
        "joblib==0.17.0",
        "pathlib==1.0.1",
        "pytsmod==0.3.3",
        #scipy==1.4.1
        #torch==1.8.0
        "tqdm==4.64.1",
        "mirdata==0.3.6",
        "essentia"
    ],
    extras_require={
        "tests": [
            "pytest>=4.4.0",
            "pytest-cov>=2.6.1",
            "pytest-pep8>=1.0.0",
            "pytest-mock>=1.10.1",
            "pytest-localserver>=0.5.0",
            "testcontainers>=2.3",
            "future==0.17.1",
            "coveralls>=1.7.0",
            "types-PyYAML",
            "types-chardet",
            "smart_open[all] >= 5.0.0",
        ],
        "docs": [
            "numpydoc",
            "recommonmark",
            "sphinx>=3.4.0",
            "sphinxcontrib-napoleon",
            "sphinx_rtd_theme",
        ],
    }
)