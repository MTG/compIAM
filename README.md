# compIAM
compIAM (**comp**utational analysis of **I**ndian **A**rt **M**usic) is a collaborative initiative involving many researchers that aims at putting together a common repository of datasets, tools, and models for the computational analysis of Carnatic and Hindustani music. 

You can get started on the Computational Analysis of Indian Art Music through our ISMIR 2022 Tutorial: [Computational Methods For Supporting Corpus-Based Research On Indian Art Music](https://github.com/MTG/IAM-tutorial-ismir22).

## Installing compIAM
compIAM is registered to PyPI, therefore the latest release can be installed with:
```
pip install compiam
```

Nonetheless, to get the latest version of the library with the fresher updates, proceed as follows:
```
git clone https://github.com/MTG/compIAM.git
cd compIAM

virtualenv -p python3 compiam_env
source compiam_env/bin/activate

pip install -e .
pip install -r requirements.txt
```

## Basic usage
### Importing the integrated tools
compIAM does not have terminal functionalities but it is to be used within Python based-projects. First, import the library to your Python project with: ``import compiam``.

The integrated tools and models are organized by the following fundamental musical aspects: melody, rhythm, structure and timbre. You can access the several included tools by importing them from their corresponding modules:
```
from compiam.melody import FTANetCarnatic
from compiam.rhythm import FourWayTabla
```

**TIP:** Print out the available tool for each category: ``compiam.melody.list_tools()``.

### Wrappers
compIAM also includes wrappers to easily initialize relevant datasets, corpora, and also pre-trained models for particular problems.

| **Wrapper**                 | **Description**                    | **Option list**                       |
|-----------------------------|------------------------------------|---------------------------------------|
| ``compiam.load_dataset()``  | Initializing dataset loaders       | Run ``compiam.list_datasets()``       |
| ``compiam.load_corpora()``  | Accessing the Dunya corpora        | Run ``compiam.list_corpora()``        |
| ``compiam.load_model()``    | Initializing pre-trained models    | Run ``compiam.list_models()``         |

## Available tools
compIAM is structured by the fundamental aspects of music in which we classify the several relevant tasks for the Indian Art Music tradition. Check here the available tools for:
- **[Melodic analysis](./compiam/melody/README.md)**
- **[Rhtyhmic analysis](./compiam/rhythm/README.md)**
- **[Structure analysis](./compiam/structure/README.md)**
- **[Timbre analysis](./compiam/timbre/README.md)**
- **Corpora access:** We do provide access to the Carnatic and Hindustani corpora in Dunya. For both corpora, there is access to the CC and the non-CC parts. More details on accessing the Dunya corpora are [given here](./compiam/dunya/README.md)
- **Datasets:** Direct access to the datasets for the computational analysis of Indian Art Music is given through [mirdata](https://github.com/mir-dataset-loaders/mirdata). The current available datasets in compIAM are:
    - [Saraga Carnatic](https://mtg.github.io/saraga/)
    - [Saraga Hindustani](https://mtg.github.io/saraga/)
    - [Mridangam Stroke Dataset](https://compmusic.upf.edu/mridangam-stroke-dataset)

## Contributing
compIAM is very much open for contributions. You can contribute by:
* Adding new datasets
* Adding tools or models
* Improving the library features
* Writing walkthroughs of the included tools
* Identifying and fixing bugs

Please check the [contribution guidelines](https://mtg.github.io/compIAM/source/contributing.html) and get in touch in case you have questions or suggestions.

## License
TODO :)

## Citing
TODO :)
