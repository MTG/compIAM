from compiam.utils import get_installed_dependencies
deps = get_installed_dependencies()

if "essentia" in deps:
    from compiam.melody.melodia import Melodia
    from compiam.melody.tonic_multipitch import TonicIndianMultiPitch
if "tensorflow" in deps:
    from compiam.melody.ftanet_carnatic import FTANetCarnatic