# compIAM
compIAM (**comp**utational analysis of **I**ndian **A**rt **M**usic) is a collaborative initiative involving many researchers that aims at putting together a common repository of datasets, tools, and models for the computational analysis of Carnatic and Hindustani music. 

## Installing compIAM
TODO :)

## Basic usage
compIAM includes wrappers to easily initialize the tools and datasets. 

| **Wrapper**                 | **Description**                    | **Option list**                       |
|-----------------------------|------------------------------------|---------------------------------------|
| ``compiam.load_dataset()``  | Initializing dataset loaders       | See ``compiam.data.datasets_list``    |
| ``compiam.load_corpora()``  | Accessing the Dunya corpora        | See ``compiam.data.corpora_list``     |
| ``compiam.load_model()``    | Initializing tools and models      | See ``compiam.data.models_dict``      |

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
* Identifying and fixing bugs

Please check the [contribution guidelines](TODO) and get in touch in case you have questions or suggestions. 
