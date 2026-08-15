
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
from LoraEasyCustomOptimizer.scorn import SCORN
from LoraEasyCustomOptimizer.scornmachina import SCORNMachina
from LoraEasyCustomOptimizer.mythical import Mythical
from LoraEasyCustomOptimizer.oagopt import OAGOpt
from LoraEasyCustomOptimizer.ocgopt import OCGOpt
from LoraEasyCustomOptimizer.glyph import Glyph
from LoraEasyCustomOptimizer.racs import RACS
from LoraEasyCustomOptimizer.alice import Alice
from LoraEasyCustomOptimizer.fira import Fira
from LoraEasyCustomOptimizer.vsgd import VSGD
from LoraEasyCustomOptimizer.cstableadamw import CStableAdamW
from LoraEasyCustomOptimizer.dehaze import Dehaze
from LoraEasyCustomOptimizer.talon import TALON
from LoraEasyCustomOptimizer.fftdescent import FFTDescent
from LoraEasyCustomOptimizer.scgopt import SCGOpt
from LoraEasyCustomOptimizer.singstate import SingState
from LoraEasyCustomOptimizer.snoo_asgd import SNOO_ASGD
from adv_optm.optim import AdamW_adv, Adopt_adv, Lion_adv, Prodigy_adv, Muon_adv, AdaMuon_adv, SignSGD_adv, SinkSGD_adv
from LoraEasyCustomOptimizer.abmog import ABMOG
from LoraEasyCustomOptimizer.bcos import BCOS
from LoraEasyCustomOptimizer.projective_adam import ProjectiveAdam
from LoraEasyCustomOptimizer.wiwiopt import WiwiOpt
from LoraEasyCustomOptimizer.adam import AdamW8bitKahan
from LoraEasyCustomOptimizer.cascade import CASCADE
from LoraEasyCustomOptimizer.radam_schedulefree import RAdamScheduleFree
from LoraEasyCustomOptimizer.nor_muon_schedulefree import NorMuonScheduleFree
from LoraEasyCustomOptimizer.ocgoptv2 import OCGOptV2
from LoraEasyCustomOptimizer.adamw_schedulefree_plus import AdamWScheduleFreePlus
from LoraEasyCustomOptimizer.amuse import AMUSE
from LoraEasyCustomOptimizer.soda import SODA
from LoraEasyCustomOptimizer.moda import MODA
from LoraEasyCustomOptimizer.soda_wrapper import SODAWrapper
from LoraEasyCustomOptimizer.bilatmuon import BilatMuon
from LoraEasyCustomOptimizer.bilatmuonns import BilatMuonNS
from LoraEasyCustomOptimizer.ainoopt import AINOOpt
from LoraEasyCustomOptimizer.warpadam import WarpAdam
from LoraEasyCustomOptimizer.warpaino import WarpAINO

OPTIMIZER_LIST: List[OPTIMIZER] = [
    ABMOG,
    AdamW8bitKahan,
    AdamWScheduleFreePlus,
    AMUSE,
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
    AdaMuon_adv,
    Adan,
    AdamW_adv,
    AdamW4bitAO,
    AdamW8bitAO,
    AdamWfp8AO,
    Adopt_adv,
    AINOOpt,
    Alice,
    BCOS,
    BilatMuon,
    BilatMuonNS,
    CAME,
    CASCADE,
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
    Lion_adv,
    MODA,
    MomentusCaution,
    Muon_adv,
    Mythical,
    NorMuonScheduleFree,
    OAGOpt,
    OCGOpt,
    OCGOptV2,
    ProdigyPlusScheduleFree,
    Prodigy_adv,
    ProjectiveAdam,
    RACS,
    REMASTER,
    RMSProp,
    RMSPropADOPT,
    RMSPropADOPTMARS,
    RAdamScheduleFree,
    Ranger21,
    SCION,
    SGDSaI,
    ScalableShampoo,
    SCGOpt,
    ScheduleFreeWrapper,
    SCORN,
    SCORNMachina,
    SimplifiedAdEMAMix,
    SimplifiedAdEMAMixExM,
    SignSGD_adv,
    SinkSGD_adv,
    SingState,
    SNOO_ASGD,
    SODA,
    SODAWrapper,
    StableSPAM,
    TALON,
    WarpAdam,
    WarpAINO,
    VSGD,
    WiwiOpt,
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}