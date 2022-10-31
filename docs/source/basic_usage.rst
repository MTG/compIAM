Basic usage
===========

Loading the tools
+++++++++++++++++

compIAM does not have terminal functionalities but it is to be used within Python based-projects. First, import the library to 
your Python project with: ``import compiam``.

The integrated tools and models are organized by the following fundamental musical aspects: melody, rhythm, structure and timbre. 
You can access the several included tools by importing them from their corresponding modules:

.. code-block:: python

    from compiam.melody import FTANetCarnatic
    from compiam.rhythm import FourWayTabla


.. note::
    Print out the available tools for each category: ``compiam.melody.list_tools()``.

Wrappers
++++++++

compIAM also includes wrappers to easily initialize relevant datasets, corpora, and also pre-trained models for particular problems.

.. autosummary::

   compiam.load_dataset
   compiam.load_corpora
   compiam.load_model