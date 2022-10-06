.. _datasets:

Load IAM datasets using mirdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a dataset in compIAM, please first open an issue in compIAM, then write
a mirdata loader for the dataset and create a PR. Once merged, please include 
then the dataset in `datasets_list` in `data.py`.


.. autofunction:: compiam.load_dataset
    

Access the Dunya corpora
^^^^^^^^^^^^^^^^^^^^^^^^

Use the Corpora class to access the Indian Art Music corpora in CompMusic.
Please note that to access the corpora in CompMusic, you need to first
register and get a personal access token. Said token will be required
to use the functions to access the database. 


.. autoclass:: compiam.dunya.Corpora
   :members:
   :inherited-members:


.. automodule:: compiam.dunya.conn
   :members:
   :inherited-members: