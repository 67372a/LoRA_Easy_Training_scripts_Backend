
from typing import Dict, List
from LoraEasyCustomOptimizer.utils import OPTIMIZER

from LoraEasyCustomOptimizer.adabelief import AdaBelief
from LoraEasyCustomOptimizer.adagc import AdaGC
from LoraEasyCustomOptimizer.adammini import AdamMini
from LoraEasyCustomOptimizer.adan import Adan
from LoraEasyCustomOptimizer.ademamix import (AdEMAMix, SimplifiedAdEMAMix, SimplifiedAdEMAMixExM)
from LoraEasyCustomOptimizer.adopt import ADOPT
from LoraEasyCustomOptimizer.came import CAME
from LoraEasyCustomOptimizer.compass import Compass, Compass8BitBNB, CompassPlus, CompassADOPT, CompassADOPTMARS, CompassAO
from LoraEasyCustomOptimizer.farmscrop import FARMSCrop, FARMSCropV2
from LoraEasyCustomOptimizer.fcompass import FCompass, FCompassPlus, FCompassADOPT, FCompassADOPTMARS
from LoraEasyCustomOptimizer.fishmonger import FishMonger, FishMonger8BitBNB
from LoraEasyCustomOptimizer.fmarscrop import FMARSCrop, FMARSCropV2, FMARSCropV2ExMachina, FMARSCropV3, FMARSCropV3ExMachina
from LoraEasyCustomOptimizer.galore import GaLore
from LoraEasyCustomOptimizer.gooddog import GOODDOG
from LoraEasyCustomOptimizer.grokfast import GrokFastAdamW
from LoraEasyCustomOptimizer.laprop import LaProp
from LoraEasyCustomOptimizer.lpfadamw import LPFAdamW
from LoraEasyCustomOptimizer.ranger21 import Ranger21
from LoraEasyCustomOptimizer.spam import StableSPAM
from LoraEasyCustomOptimizer.rmsprop import RMSProp, RMSPropADOPT, RMSPropADOPTMARS
from LoraEasyCustomOptimizer.schedulefree import (
    ScheduleFreeWrapper, ADOPTScheduleFree, ADOPTEMAMixScheduleFree, ADOPTNesterovScheduleFree, 
    FADOPTScheduleFree, ADOPTMARSScheduleFree, FADOPTMARSScheduleFree, ADOPTAOScheduleFree
    )

from LoraEasyCustomOptimizer.clybius_experiments import (MomentusCaution, REMASTER)
from LoraEasyCustomOptimizer.scion import SCION
from LoraEasyCustomOptimizer.sgd import SGDSaI
from LoraEasyCustomOptimizer.shampoo import ScalableShampoo
from LoraEasyCustomOptimizer.adam import AdamW8bitAO, AdamW4bitAO, AdamWfp8AO
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from .prodigy_plus.prodigy_plus_schedulefree import ProdigyPlusExMachinaScheduleFree
from LoraEasyCustomOptimizer.scorn import SCORN
from LoraEasyCustomOptimizer.scornmachina import SCORNMachina
from LoraEasyCustomOptimizer.mythical import Mythical
from LoraEasyCustomOptimizer.glyph import Glyph
from LoraEasyCustomOptimizer.racs import RACS
from LoraEasyCustomOptimizer.alice import Alice
from LoraEasyCustomOptimizer.fira import Fira
from LoraEasyCustomOptimizer.vsgd import VSGD
from LoraEasyCustomOptimizer.cstableadamw import CStableAdamW
from LoraEasyCustomOptimizer.dehaze import Dehaze
from LoraEasyCustomOptimizer.talon import TALON
from LoraEasyCustomOptimizer.fftdescent import FFTDescent
from LoraEasyCustomOptimizer.singstate import SingState

OPTIMIZER_LIST: List[OPTIMIZER] = [
    ADOPT,
    ADOPTAOScheduleFree,
    ADOPTEMAMixScheduleFree,
    ADOPTMARSScheduleFree,
    ADOPTNesterovScheduleFree,
    ADOPTScheduleFree,
    AdEMAMix,
    AdaBelief,
    AdaGC,
    AdamMini,
    Adan,
    AdamW4bitAO,
    AdamW8bitAO,
    AdamWfp8AO,
    Alice,
    CAME,
    Compass,
    CompassAO,
    Compass8BitBNB,
    CompassADOPT,
    CompassADOPTMARS,
    CompassPlus,
    CStableAdamW,
    Dehaze,
    FADOPTMARSScheduleFree,
    FADOPTScheduleFree,
    FARMSCrop,
    FARMSCropV2,
    FCompass,
    FCompassADOPT,
    FCompassADOPTMARS,
    FCompassPlus,
    Fira,
    FMARSCrop,
    FMARSCropV2,
    FMARSCropV2ExMachina,
    FMARSCropV3,
    FMARSCropV3ExMachina,
    FishMonger,
    FishMonger8BitBNB,
    FFTDescent,
    GaLore,
    Glyph,
    GOODDOG,
    GrokFastAdamW,
    LPFAdamW,
    LaProp,
    MomentusCaution,
    Mythical,
    ProdigyPlusScheduleFree,
    ProdigyPlusExMachinaScheduleFree,
    RACS,
    REMASTER,
    RMSProp,
    RMSPropADOPT,
    RMSPropADOPTMARS,
    Ranger21,
    SCION,
    SGDSaI,
    ScalableShampoo,
    ScheduleFreeWrapper,
    SCORN,
    SCORNMachina,
    SimplifiedAdEMAMix,
    SimplifiedAdEMAMixExM,
    SingState,
    StableSPAM,
    TALON,
    VSGD,
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}