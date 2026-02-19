.. _melody:

Melodic analysis
================

Tonic Identification
++++++++++++++++++++

Tonic Indian Art Music (Multipitch approach)
--------------------------------------------

.. note::
    REQUIRES: essentia

.. automodule:: compiam.melody.tonic_identification.tonic_multipitch
   :members:


Pitch Extraction
++++++++++++++++

Melodia
-------

.. note::
    REQUIRES: essentia

.. automodule:: compiam.melody.pitch_extraction.melodia
   :members:

FTANet-Carnatic
---------------

.. note::
    REQUIRES: tensorflow

.. autoclass:: compiam.melody.pitch_extraction.FTANetCarnatic
   :members:

FTAResNet-Carnatic
------------------

.. note::
    REQUIRES: torch

.. autoclass:: compiam.melody.pitch_extraction.FTAResNetCarnatic
   :members:


Melodic Pattern Discovery
+++++++++++++++++++++++++

CAE-Carnatic (Wrapper)
----------------------

.. note::
    REQUIRES: torch

.. autoclass:: compiam.melody.pattern.sancara_search.CAEWrapper
   :members:

Self-similarity matrix
----------------------

.. note::
    REQUIRES: torch

.. automodule:: compiam.melody.pattern.sancara_search.extraction.self_sim
   :members:


Raga Recognition
++++++++++++++++

DEEPSRGM
--------

.. note::
    REQUIRES: torch

.. autoclass:: compiam.melody.raga_recognition.deepsrgm.DEEPSRGM
   :members:
