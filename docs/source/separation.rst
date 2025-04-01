.. _separation:

Separation
==========

Singing voice extraction
++++++++++++++++++++++++

Leakage-aware Carnatic Singing Voice Separation
-----------------------------------------------

.. note::
    REQUIRES: tensorflow

.. autoclass:: compiam.separation.singing_voice_extraction.cold_diff_sep.ColdDiffSep
   :members:


Leakage-aware Carnatic Singing Voice Separation
-----------------------------------------------

.. note::
    REQUIRES: torch

.. autoclass:: compiam.separation.singing_voice_extraction.convtdf_vocal_finetune.ConvTDFVocalFineTune
   :members:


Vocals and violin separation
++++++++++++++++++++++++++++

MDXNet mixer model to separate vocals and violin 
------------------------------------------------

.. note::
    REQUIRES: torch

.. autoclass:: compiam.separation.music_source_separation.mixer_model.MixerModel
   :members: