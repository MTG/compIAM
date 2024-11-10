# compIAM v0.3.0
compIAM (**comp**utational analysis of **I**ndian **A**rt **M**usic) is a collaborative initiative involving many researchers that aims at putting together a common repository of datasets, tools, and models for the computational analysis of Carnatic and Hindustani music. 

You can get started on the Computational Analysis of Indian Art Music through our ISMIR 2022 Tutorial: [Computational Methods For Supporting Corpus-Based Research On Indian Art Music](https://mtg.github.io/IAM-tutorial-ismir22/landing.html).


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

**Python version:** At this moment, we have successfully tested ``compiam`` in ``python`` versions: ``3.9 - 3.11``. Support to ``py3.8`` has been dropped, since it reached end-of-life. Support to ``3.12`` and ``3.13`` is coming soon!


## Basic usage
### Importing the integrated tools
compIAM does not have terminal functionalities but it is to be used within Python based-projects. First, import the library to your Python project with: ``import compiam``.

The integrated tools and models are organized by:
1) The following fundamental musical aspects: *melody*, *rhythm*, *structure* and *timbre* (in v0.3.0 we are introducing new section, *separation*)
2) The task these tools tackle.

You can access the several included tools by importing them from their corresponding modules:
```
from compiam.melody.pitch_extraction import FTANetCarnatic
from compiam.rhythm.transcription import FourWayTabla
```

**TIP:** Print out the available tool for each category: ``compiam.melody.list_tools()``. Print out the available tasks for each category: ``compiam.melody.list_tasks()``, and print out the available tools for each module using: ``compiam.melody.list_tools()``. You may also list only the tools for a particular task: ``compiam.melody.pitch_extraction.list_tools()``

### Wrappers
compIAM also includes wrappers to easily initialize relevant datasets, corpora, and also pre-trained models for particular problems.

| **Wrapper**                 | **Description**                    | **Option list**                       |
|-----------------------------|------------------------------------|---------------------------------------|
| ``compiam.load_dataset()``  | Initializing dataset loaders       | Run ``compiam.list_datasets()``       |
| ``compiam.load_corpora()``  | Accessing the Dunya corpora        | Run ``compiam.list_corpora()``        |
| ``compiam.load_model()``    | Initializing pre-trained models    | Run ``compiam.list_models()``         |


## Available components
### Tools and models
compIAM is structured by the fundamental aspects of music in which we classify the several relevant tasks for the Indian Art Music tradition. Check here the available tools for:
- **[Melodic analysis](./compiam/melody/README.md)**
- **[Rhtyhmic analysis](./compiam/rhythm/README.md)**
- **[Structure analysis](./compiam/structure/README.md)**
- **[Timbre analysis](./compiam/timbre/README.md)**
- **[Music source separation](./compiam/separation/README.md)**

### Accessing the Dunya corpora
We do provide access to the Carnatic and Hindustani corpora in Dunya. For both corpora, there is access to the CC and the non-CC parts. More details on accessing the Dunya corpora are [given here](./compiam/dunya/README.md). 

### Dataset loaders
Direct and MIR-standardized access to the datasets for the computational analysis of Indian Art Music is given through [mirdata](https://github.com/mir-dataset-loaders/mirdata) loaders. The current available datasets in compIAM are:
* [Saraga Carnatic](https://mtg.github.io/saraga/)*
* [Saraga Hindustani](https://mtg.github.io/saraga/)*
* [Indian Art Music Raga Dataset](https://zenodo.org/record/7278506)*
* [Indian Art Music Tonic Dataset](https://zenodo.org/record/1257114)*
* [Carnatic Music Rhythm](https://zenodo.org/record/1264394)*
* [Hindustani Music Rhythm](https://zenodo.org/record/1264742)*
* [Mridangam Stroke Dataset](https://compmusic.upf.edu/mridangam-stroke-dataset)*
* [Carnatic Varnam](https://zenodo.org/records/7726167)*
* [Four-Way Tabla Stroke (ISMIR 21)](https://zenodo.org/record/7110248)
* [SCMS (Saraga Carnatic Melody Synth)](https://zenodo.org/records/5553925)

The datasets marked with * have been compiled within the framework of the [CompMusic project](https://compmusic.upf.edu/).


## Contributing
compIAM is very much open for contributions. You can contribute by:
* Adding new datasets
* Adding tools or models
* Improving the library features
* Writing walkthroughs of the included tools
* Identifying and fixing bugs

Please check the [contribution guidelines](https://mtg.github.io/compIAM/source/contributing.html) and get in touch in case you have questions or suggestions.


## Example notebooks
We include, in this repo, [example notebooks](https://github.com/MTG/compIAM/tree/master/notebooks) for users to better understand how to use `compiam` and also showcase 

## License
compIAM is Copyright 2024 Music Technology Group - Universitat Pompeu Fabra

compIAM is released under the terms of the GNU Affero General Public License (v3 or later). See the COPYING file for more information. For the case of a particular tool or implementation that has a specific different licence, this is explicitly specified in the files related to this tool, and these terms must be followed.

For any licensing enquires, please contact us at [mtg-info@upf.edu](mailto:mtg-info@upf.edu)


## Citing
```bibtex
@software{compiam_mtg_2023,
  author = {{Genís Plaja-Roglans and Thomas Nuttall and Xavier Serra}},
  title = {compIAM},
  url = {https://mtg.github.io/compIAM/},
  version = {0.3.0},
  year = {2023}
}
```
