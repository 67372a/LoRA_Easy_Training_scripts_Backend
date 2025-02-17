
from typing import Dict, List
from LoraEasyCustomOptimizer.utils import OPTIMIZER

from LoraEasyCustomOptimizer.adabelief import AdaBelief
from LoraEasyCustomOptimizer.adammini import AdamMini
from LoraEasyCustomOptimizer.adan import Adan
from LoraEasyCustomOptimizer.ademamix import AdEMAMix
from LoraEasyCustomOptimizer.adopt import ADOPT
from LoraEasyCustomOptimizer.came import CAME
from LoraEasyCustomOptimizer.compass import Compass, Compass8BitBNB, CompassPlus, CompassADOPT, CompassADOPTMARS, CompassAO
from LoraEasyCustomOptimizer.farmscrop import FARMSCrop, FARMSCropV2
from LoraEasyCustomOptimizer.fcompass import FCompass, FCompassPlus, FCompassADOPT, FCompassADOPTMARS
from LoraEasyCustomOptimizer.fishmonger import FishMonger, FishMonger8BitBNB
from LoraEasyCustomOptimizer.fmarscrop import FMARSCrop, FMARSCropV2, FMARSCropV2ExMachina, FMARSCropV3, FMARSCropV3ExMachina
from LoraEasyCustomOptimizer.galore import GaLore
from LoraEasyCustomOptimizer.grokfast import GrokFastAdamW
from LoraEasyCustomOptimizer.laprop import LaProp
from LoraEasyCustomOptimizer.lpfadamw import LPFAdamW
from LoraEasyCustomOptimizer.ranger21 import Ranger21
from LoraEasyCustomOptimizer.rmsprop import RMSProp, RMSPropADOPT, RMSPropADOPTMARS
from LoraEasyCustomOptimizer.schedulefree import (
    ScheduleFreeWrapper, ADOPTScheduleFree, ADOPTEMAMixScheduleFree, ADOPTNesterovScheduleFree, 
    FADOPTScheduleFree, ADOPTMARSScheduleFree, FADOPTMARSScheduleFree, ADOPTAOScheduleFree
    )

from LoraEasyCustomOptimizer.clybius_experiments import MomentusCaution
from LoraEasyCustomOptimizer.sgd import SGDSaI
from LoraEasyCustomOptimizer.shampoo import ScalableShampoo
from LoraEasyCustomOptimizer.adam import AdamW8bitAO, AdamW4bitAO, AdamWfp8AO
from .distributed_shampoo.distributed_shampoo import DistributedShampoo
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from .prodigy_plus.prodigy_plus_schedulefree import ProdigyPlusExMachinaScheduleFree

OPTIMIZER_LIST: List[OPTIMIZER] = [
    ADOPT,
    ADOPTAOScheduleFree,
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
    AdamWfp8AO,
    CAME,
    Compass,
    CompassAO,
    Compass8BitBNB,
    CompassADOPT,
    CompassADOPTMARS,
    CompassPlus,
    DistributedShampoo,
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
    FMARSCropV3,
    FMARSCropV3ExMachina,
    FishMonger,
    FishMonger8BitBNB,
    GaLore,
    GrokFastAdamW,
    LPFAdamW,
    LaProp,
    MomentusCaution,
    ProdigyPlusScheduleFree,
    ProdigyPlusExMachinaScheduleFree,
    RMSProp,
    RMSPropADOPT,
    RMSPropADOPTMARS,
    Ranger21,
    SGDSaI,
    ScalableShampoo,
    ScheduleFreeWrapper,
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}