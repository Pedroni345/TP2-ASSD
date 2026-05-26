from .base import AudioEffect, EffectChain
from .reverb import Echo, Reverb, MultiTapReverb, ConvolutionReverb
from .distortion import Clipper, SoftClipper
from .flanger import Flanger, Vibrato, Chorus
from .phaser import Phaser
from .wahwah import WahWah


EFFECT_REGISTRY: dict[str, type] = {
    "Echo": Echo,
    "Reverb": Reverb,
    "Multi-Tap Reverb": MultiTapReverb,
    "Conv. Reverb": ConvolutionReverb,
    "Clipping": Clipper,
    "Soft Clip": SoftClipper,
    "Flanger": Flanger,
    "Vibrato": Vibrato,
    "Chorus": Chorus,
    "Phaser": Phaser,
    "Wah-Wah": WahWah,
}
