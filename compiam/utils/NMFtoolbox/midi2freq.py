"""
    Name: midi2freq
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""


def midi2freq(midi):
    """Converts a given MIDI pitch to the corresponding frequency in Hz. No
    sanity checks on the validity of the input are performed.

    Parameters
    ----------
    midi: array-like / float
        The MIDI pitch, can also be floating value

    Returns
    -------
    freq: array-like / float
        The frequency in Hz
    """
    freq = (440.0 / 32) * 2 ** ((midi - 9) / 12)

    return freq