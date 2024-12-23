
from typing import Dict, List
from LoraEasyCustomOptimizer.utils import OPTIMIZER

from LoraEasyCustomOptimizer.adabelief import AdaBelief
from LoraEasyCustomOptimizer.adammini import AdamMini
from LoraEasyCustomOptimizer.adan import Adan
from LoraEasyCustomOptimizer.ademamix import AdEMAMix
from LoraEasyCustomOptimizer.adopt import ADOPT
from LoraEasyCustomOptimizer.came import CAME
from LoraEasyCustomOptimizer.compass import Compass, Compass8BitBNB, CompassPlus, CompassADOPT, CompassADOPTMARS
from LoraEasyCustomOptimizer.farmscrop import FARMSCrop, FARMSCropV2
from LoraEasyCustomOptimizer.fcompass import FCompass, FCompassPlus, FCompassADOPT, FCompassADOPTMARS
from LoraEasyCustomOptimizer.fishmonger import FishMonger, FishMonger8BitBNB
from LoraEasyCustomOptimizer.fmarscrop import FMARSCrop, FMARSCropV2, FMARSCropV2ExMachina
from LoraEasyCustomOptimizer.galore import GaLore
from LoraEasyCustomOptimizer.grokfast import GrokFastAdamW
from LoraEasyCustomOptimizer.laprop import LaProp
from LoraEasyCustomOptimizer.lpfadamw import LPFAdamW
from LoraEasyCustomOptimizer.ranger21 import Ranger21
from LoraEasyCustomOptimizer.rmsprop import RMSProp, RMSPropADOPT, RMSPropADOPTMARS
from LoraEasyCustomOptimizer.schedulefree import ScheduleFreeWrapper, ADOPTScheduleFree, ADOPTEMAMixScheduleFree, ADOPTNesterovScheduleFree, FADOPTScheduleFree, ADOPTMARSScheduleFree, FADOPTMARSScheduleFree
from LoraEasyCustomOptimizer.sgd import SGDSaI
from LoraEasyCustomOptimizer.shampoo import ScalableShampoo
from LoraEasyCustomOptimizer.adam import AdamW8bitAO, AdamW4bitAO

OPTIMIZER_LIST: List[OPTIMIZER] = [
    ADOPT,
    ADOPTEMAMixScheduleFree,
    ADOPTMARSScheduleFree,
    ADOPTNesterovScheduleFree,
    ADOPTScheduleFree,
    AdEMAMix,
    AdaBelief,
    AdamMini,
    Adan,
    AdamW4bitAO,
    AdamW8bitAO,
    CAME,
    Compass,
    Compass8BitBNB,
    CompassADOPT,
    CompassADOPTMARS,
    CompassPlus,
    FADOPTMARSScheduleFree,
    FADOPTScheduleFree,
    FARMSCrop,
    FARMSCropV2,
    FCompass,
    FCompassADOPT,
    FCompassADOPTMARS,
    FCompassPlus,
    FMARSCrop,
    FMARSCropV2,
    FMARSCropV2ExMachina,
    FishMonger,
    FishMonger8BitBNB,
    GaLore,
    GrokFastAdamW,
    LPFAdamW,
    LaProp,
    RMSProp,
    RMSPropADOPT,
    RMSPropADOPTMARS,
    Ranger21,
    SGDSaI,
    ScalableShampoo,
    ScheduleFreeWrapper,
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}