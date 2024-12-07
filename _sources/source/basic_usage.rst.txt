Basic usage
===========

Loading the tools
+++++++++++++++++

compIAM does not have terminal functionalities but it is to be used within Python based-projects. First, import the library to 
your Python project with: ``import compiam``.

The integrated tools and models are organized by:

#. First, the following fundamental musical aspects: melody, rhythm, structure, timbre, and from v0.3.0 we have included separation as well.
#. Then, the tools are grouped by tasks.

You can access the several included tools by importing them from their corresponding modules:

.. code-block:: python

    from compiam.melody.pitch_extraction import FTANetCarnatic
    from compiam.rhythm.transcription import FourWayTabla


We provide nice functionalities to explore where do the tools live: 

#. Print out the available tasks for each category: ``compiam.melody.list_tasks()``
#. Print out the available tools for each module using: ``compiam.melody.list_tools()``
#. Print out only the tools for a particular task: ``compiam.melody.pitch_extraction.list_tools()``

.. important::
    Some tools (especially the ML/DL models) require specific dependencies that are not installed by default, 
    because of their size or compatibility issues. If a tool is loaded and a particular dependency is missing, 
    an alert will be displayed, to inform the user which dependency is missing and how to proceed to install
    it in the right version. See ``optional_requirements.txt`` where the optional dependencies and
    the specific versions we use in ``compiam`` are listed.


Wrappers
++++++++

compIAM also includes wrappers to easily initialize relevant datasets, corpora, and also pre-trained models for particular problems.

.. autosummary::

   compiam.load_dataset
   compiam.load_corpora
   compiam.load_model


.. tip::
    When listing available tools using the ``list_tools()`` functions, some will appear with a "*" at the end. That is meant to 
    indicate that such tools have pre-trained models available, which may be loaded using the wrapper ``compiam.load_model()``.

Model weights are large in size and therefore, not included in the library from v0.2.1 on. We have included a ``.download_model()``
function to all ML/DL models that require pre-trained weights, so that the user can download them on demand. This function is
automatically run when invoking the model through the ``compiam.load_model()`` wrapper. The model weights are then stored in the
corresponding default folder ``./compiam/model/``. If the model is already downloaded, the function will not download it again.

.. note::
    From v0.3.0 on, ``compiam.load_model()`` wrapper has an argument ``data_home``, in where you can specify to which folder you
    want the models to be downloaded and read from.

.. note::
    From v0.4.0 on, ``compiam.load_model()`` wrapper has an argument ``version``, in where you can specify which version of the 
    pre-trained model you want to use. You may want to use ``compiam.get_model_info(<model_key>)`` to print out the entire model
    information in compiam/data.py and visualisse the available versions. By default, the model contributor selects a default version
    to be loaded without the user having to specify it, so this argument is option. If you try to load a non-existing version, an 
    error will be thrown.