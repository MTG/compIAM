.. _contributing guidelines:

Contributing pipeline
^^^^^^^^^^^^^^^^^^^^^

Integrating a tool
------------------
We will present here the necessary steps to follow to integrate a tool into ``compiam``:

* Open an issue to ``compiam`` lettins us know that you are integating a tool, so that we can be aware of that and rapidly provide support if needed.

* Create a ``tool_name.py`` or ``tool_name/`` within the most appropiate ``compiam/<musical-aspect>/<task>/``. Important considerations:

    * If you are proposing a new task, you can also create a new task folder, copying and adapting the ``__init__.py`` file from another task.

    * The decision between ``tool_name.py`` vs ``tool_name/`` depends on how complex your tool is. If your tool is composed by multiple files, you can use the folder for a cleaner organization.  However, you will see that some tools in ``compiam`` can be contained in a single file.

    * Your tool should have a ``.extract()`` method in case it is knowledge-based, or a ``.predict()`` model in case it is data-driven.

* Make sure that after integrating the tool, you add it to the ``__init__.py`` file of the corresponding task. Make sure also to add the citation in the README file in the musical aspect.

* If your model is ML/DL-based and you want to allow users to load a provided pre-train weights, please include these into ``compiam/models/<musical-aspect>`` and add the model into ``models_dict`` in ``data.py``.

The case of *optional dependencies*:

* If your tool uses one of these dependencies: ``tensorflow``, ``torch``, ``essentia``, or you think that
one of your dependencies might be added as optional, please reach out to us, and we can discuss how 
we address it. You can follow one of the already implemented tools that include optional dependencies
as an example. We follow a very specific method to handle this situations and we need to continue 
consistently.

Integrating a dataset
---------------------
We use ``mirdata`` a backbone for dataset management. Therefore, in order to integate a dataset, please
refer to the ``mirdata`` repository where very clear contribution guidelines are provided.