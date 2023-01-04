.. _datasets:

Load IAM datasets using mirdata
===============================

compIAM includes an alias function for ``mirdata.initialize()`` to directly
initialize the mirdata loaders of Indian Art Music datasets.

.. note::
    Print out the available datasets to load: ``compiam.list_datasets()``.

.. autofunction:: compiam.load_dataset
    

Access the Dunya corpora
========================

Use the Corpora class to access the Indian Art Music corpora in CompMusic.
Please note that to access the corpora in CompMusic, you need to first
register and get a personal access token. Said token will be required
to use the functions to access the database. 


.. automodule:: compiam.dunya
   :members:


.. automodule:: compiam.dunya.conn
   :members: