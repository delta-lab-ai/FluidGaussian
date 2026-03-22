from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .V_sel import VarSelector
from .combine_selector import CombSelector
from .physic_selector import SimSelector
from .combine_active_selector import CombActiveSelector
methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "variance": VarSelector, "combine": CombSelector, "physics": SimSelector, "combine_active": CombActiveSelector}