import os
from joblib import Parallel, delayed

from compiam.utils import create_if_not_exists
from compiam.utils.augment.augment_data_ar import augment_data_attack_remix

# from compiam.utils.augment.augment_data_ps import augment_data_pitch_shift
from compiam.utils.augment.augment_data_sf import augment_data_spectral_shape
from compiam.utils.augment.augment_data_sr import augment_data_stroke_remix

# from compiam.utils.augment.augment_data_ts import augment_data_time_scale

file_dir = os.path.dirname(__file__)


# def pitch_shift(in_path, out_dir, shifts=-1.0, sr=16000, n_jobs=4):
#    """Pitch shift audio at <in_path>, by shifts in <shifts>. Output to <out_dir>
#
#    :param in_path: Path to input audio
#    :type in_path: str
#    :param out_dir: Directory to output pitch shifted audios
#    :type out_dir: str
#    :param shifts: Pitch shifts value in semitones (or list of values for multiple outputs)
#    :type shifts: float or list
#    :param sr: Sampling rate of input audio
#    :type sr: float
#    :param n_jobs: n jobs for parrelelization
#    :type n_jobs: int
#    """
#    create_if_not_exists(out_dir)
#
#    if not isinstance(shifts, list):
#        shifts = [shifts]
#
#    _ = Parallel(n_jobs=n_jobs)(
#        delayed(augment_data_pitch_shift)(in_path, out_dir, sr, ps) for ps in shifts
#    )


def spectral_shape(
    in_path,
    out_dir,
    gain_factors=(0.6, 2, 0.2),
    winDur=46.4,
    hopDur=5e-3,
    sr=16000,
    n_jobs=4,
):
    """Augmenting data by perturbin 'nuisance attributes' that
    are unimportant in the specific discrimination task.

    :param in_path: Path to input audio
    :type in_path: str
    :param out_dir: Directory to output pitch shifted audios
    :type out_dir: str
    :param gain_factors: 3-tuple (or list of) with gain factors for remixing. Tuple entries correspond to each of bass, treble, & damped components.
    :type gain_factors: float or list
    :param winDur: Window size in milliseconds
    :type winDur: float
    :param hopDur: Hop size in milliseconds
    :type hopDur: float
    :param sr: Sampling rate of input audio
    :type sr: float
    :param n_jobs: n jobs for parrelelization
    :type n_jobs: int
    """
    create_if_not_exists(out_dir)

    if not isinstance(gain_factors, list):
        gain_factors = [gain_factors]

    _ = Parallel(n_jobs=n_jobs)(
        delayed(augment_data_spectral_shape)(
            in_path, out_dir, sr, gain_set, winDur=winDur, hopDur=hopDur
        )
        for gain_set in gain_factors
    )


def stroke_remix(
    in_path,
    out_dir,
    gain_factors=(0.6, 2, 0.2),
    templates=os.path.join(file_dir, "augmentation", "templates.npy"),
    winDur=46.4,
    hopDur=5e-3,
    sr=16000,
    n_jobs=4,
):
    """Simulate the expected variations of relative strengths of drums in a mix
     using non-negative matrix factorization (NMF).

    :param in_path: Path to input audio
    :type in_path: str
    :param out_dir: Directory to output pitch shifted audios
    :type out_dir: str
    :param gain_factors: 3-tuple (or list of) with gain factors for remixing. Tuple entries correspond to each of bass, treble, & damped components.
    :type gain_factors: float or list
    :param templates: path to saved nmf templates
    :type templates: str
    :param winDur: Window size in milliseconds
    :type winDur: float
    :param hopDur: Hop size in milliseconds
    :type hopDur: float
    :param sr: Sampling rate of input audio
    :type sr: float
    :param n_jobs: n jobs for parrelelization
    :type n_jobs: int
    """
    create_if_not_exists(out_dir)

    if not isinstance(gain_factors, list):
        gain_factors = [gain_factors]

    _ = Parallel(n_jobs=n_jobs)(
        delayed(augment_data_stroke_remix)(
            in_path, out_dir, sr, gain_set, templates, winDur, hopDur
        )
        for gain_set in gain_factors
    )


# def time_scale(in_path, out_dir, time_shifts=0.8, sr=16000, n_jobs=4):
#    """Time scaling of input audio, pitch maintained.
#
#    :param in_path: Path to input audio
#    :type in_path: str
#    :param out_dir: Directory to output pitch shifted audios
#    :type out_dir: str
#    :param time_shifts: time scale value (or list of)
#    :type time_shifts: float or list
#    :param sr: Sampling rate of input audio
#    :type sr: float
#    :param n_jobs: n jobs for parrelelization
#    :type n_jobs: int
#    """
#    create_if_not_exists(out_dir)
#
#    if not isinstance(time_shifts, list):
#        time_shifts = [time_shifts]
#
#    _ = Parallel(n_jobs=n_jobs)(
#        delayed(augment_data_time_scale)(in_path, out_dir, sr, ts) for ts in time_shifts
#    )


def attack_remix(
    in_path, out_dir, gain_factors=0.3, winDur=46.4, hopDur=5, sr=16000, n_jobs=4
):
    """Modifying the relative levels of attack and decay regions of an audio

    :param in_path: Path to input audio
    :type in_path: str
    :param out_dir: Directory to output pitch shifted audios
    :type out_dir: str
    :param gain_factors: gain factor (or list of) to scale attack portion with
    :type gain_factors: float or list
    :param winDur: Window size in milliseconds
    :type winDur: float
    :param hopDur: Hop size in milliseconds
    :type hopDur: float
    :param sr: Sampling rate of input audio
    :type sr: float
    :param n_jobs: n jobs for parrelelization
    :type n_jobs: int
    """
    create_if_not_exists(out_dir)

    if not isinstance(gain_factors, list):
        gain_factors = [gain_factors]

    _ = Parallel(n_jobs=n_jobs)(
        delayed(augment_data_attack_remix)(in_path, out_dir, sr, G, winDur, hopDur)
        for G in gain_factors
    )
