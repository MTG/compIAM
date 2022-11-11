Welcome to compiam's documentation!
===================================

compIAM is a collaborative initiative lead by the MTG and involving many researchers that aims at putting together data 
loaders, tools, and models for the computational analysis of two main Indian Art Music traditions: Carnatic and Hindustani.

Installing compIAM
++++++++++++++++++

compIAM is registered to PyPI, therefore the latest release can be installed with:

.. code-block:: bash

    pip install compiam


Nonetheless, to get the latest version of the library with the fresher updates, proceed as follows:

.. code-block:: bash

    git clone https://github.com/MTG/compIAM.git
    cd compIAM

    virtualenv -p python3 compiam_env
    source compiam_env/bin/activate

    pip install -e .
    pip install -r requirements.txt


Citation
++++++++
If you use compIAM for your research, please consider citing our work as:

.. code-block:: bibtex

    @software{compiam_mtg_2022,
        author = {{Genís Plaja-Roglans and Thomas Nuttall and Xavier Serra}},
        title = {compIAM},
        url = {https://mtg.github.io/compIAM/},
        version = {0.1.0},
        year = {2022}
    }


.. toctree::
   :caption: Basic usage
   :hidden:

   source/basic_usage

.. toctree::
   :caption: Tools and models
   :hidden:

   source/melody
   source/rhythm
   source/structure
   source/timbre

.. toctree::
   :caption: Datasets
   :hidden:

   source/datasets

.. toctree::
   :caption: Miscellaneous
   :hidden:

   source/visualisation
   source/utils

.. toctree::
   :caption: Contributing
   :hidden:

   source/contributing