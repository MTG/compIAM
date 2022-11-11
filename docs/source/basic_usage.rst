Basic usage
===========

Loading the tools
+++++++++++++++++

compIAM does not have terminal functionalities but it is to be used within Python based-projects. First, import the library to 
your Python project with: ``import compiam``.

The integrated tools and models are organized by:
1) The following fundamental musical aspects: melody, rhythm, structure and timbre. 
2) The task these tools tackle.

You can access the several included tools by importing them from their corresponding modules:

.. code-block:: python

    from compiam.melody.pitch_extraction import FTANetCarnatic
    from compiam.rhythm.transcription import FourWayTabla


.. note::
    Print out the available tasks for each category: ``compiam.melody.list_tasks()``, 
    and print out the available tools for each module using: ``compiam.melody.list_tools()``.
    You may also list only the tools for a particular task: ``compiam.melody.pitch_extraction.list_tools()``

Make sure to check out the documentation to identify how the package is organized and where does each tool lives.


Wrappers
++++++++

compIAM also includes wrappers to easily initialize relevant datasets, corpora, and also pre-trained models for particular problems.

.. autosummary::

   compiam.load_dataset
   compiam.load_corpora
   compiam.load_model