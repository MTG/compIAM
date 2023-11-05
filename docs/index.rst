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


License
+++++++
compIAM is Copyright 2022 Music Technology Group - Universitat Pompeu Fabra

compIAM is released under the terms of the GNU Affero General Public License (v3 or later). 
See the COPYING file for more information. For the case of a particular tool or implementation 
that has a specific different licence, this is explicitly specified in the files related to this
tool, and these terms must be followed.

For any licensing enquires, please contact us at `mtg-info@upf.edu`_.

.. _mtg-info@upf.edu: mailto:mtg-info@upf.edu


Citation
++++++++
If you use compIAM for your research, please consider citing our work as:

.. code-block:: bibtex

    @software{compiam_mtg,
        author = {{Genís Plaja-Roglans and Thomas Nuttall and Xavier Serra}},
        title = {compIAM},
        url = {https://mtg.github.io/compIAM/},
        version = {0.3.0},
        year = {2023}
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
   source/separation

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