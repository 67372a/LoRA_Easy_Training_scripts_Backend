
from typing import Dict, List
from LoraEasyCustomOptimizer.utils import OPTIMIZER

from LoraEasyCustomOptimizer.ademamix import AdEMAMix
from LoraEasyCustomOptimizer.came import CAME
from LoraEasyCustomOptimizer.compass import Compass, Compass8BitBNB, CompassPlus, CompassADOPT, CompassADOPTMARS
from LoraEasyCustomOptimizer.farmscrop import FARMSCrop, FARMSCropV2
from LoraEasyCustomOptimizer.fcompass import FCompass, FCompassPlus, FCompassADOPT, FCompassADOPTMARS
from LoraEasyCustomOptimizer.fishmonger import FishMonger, FishMonger8BitBNB
from LoraEasyCustomOptimizer.lpfadamw import LPFAdamW
from LoraEasyCustomOptimizer.rmsprop import RMSProp, RMSPropADOPT, RMSPropADOPTMARS
from LoraEasyCustomOptimizer.shampoo import ScalableShampoo
from LoraEasyCustomOptimizer.soap import SOAP
from LoraEasyCustomOptimizer.ranger21 import Ranger21
from LoraEasyCustomOptimizer.lamb import Lamb
from LoraEasyCustomOptimizer.adan import Adan
from LoraEasyCustomOptimizer.sam import SAM, GSAM, WSAM, BSAM
from LoraEasyCustomOptimizer.adopt import ADOPT
from LoraEasyCustomOptimizer.grokfast import GrokFastAdamW
from LoraEasyCustomOptimizer.adammini import AdamMini
from LoraEasyCustomOptimizer.adai import Adai
from LoraEasyCustomOptimizer.adabelief import AdaBelief
from LoraEasyCustomOptimizer.galore import GaLore
from LoraEasyCustomOptimizer.schedulefree import ScheduleFreeWrapper, ADOPTScheduleFree, ADOPTEMAMixScheduleFree, ADOPTNesterovScheduleFree, FADOPTScheduleFree, ADOPTMARSScheduleFree, FADOPTMARSScheduleFree
from LoraEasyCustomOptimizer.fmarscrop import FMARSCrop, FMARSCropV2, FMARSCropV2ExMachina
from LoraEasyCustomOptimizer.laprop import LaProp
from LoraEasyCustomOptimizer.sgd import SGDSaI

OPTIMIZER_LIST: List[OPTIMIZER] = [
    AdEMAMix,
    CAME,
    Compass,
    Compass8BitBNB,
    FARMSCrop,
    FCompass,
    FishMonger,
    FishMonger8BitBNB,
    LPFAdamW,
    RMSProp,
    ScalableShampoo,
    SOAP,
    Ranger21,
    CompassPlus,
    FCompassPlus,
    Lamb,
    Adan,
    SAM,
    GSAM,
    WSAM,
    BSAM,
    ADOPT,
    GrokFastAdamW,
    AdamMini,
    Adai,
    AdaBelief,
    GaLore,
    ScheduleFreeWrapper,
    FARMSCropV2,
    FMARSCrop,
    LaProp,
    ADOPTScheduleFree,
    ADOPTEMAMixScheduleFree,
    ADOPTNesterovScheduleFree,
    FADOPTScheduleFree,
    ADOPTMARSScheduleFree,
    FADOPTMARSScheduleFree,
    CompassADOPT,
    RMSPropADOPT,
    CompassADOPTMARS,
    RMSPropADOPTMARS,
    FMARSCropV2,
    SGDSaI,
    FCompassADOPT,
    FCompassADOPTMARS,
    FMARSCropV2ExMachina,
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}