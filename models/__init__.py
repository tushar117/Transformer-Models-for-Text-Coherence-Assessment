from .hierarchical import HierarchicalModel
from .vanilla import TransformerModel
from .fact_aware import FactAwareModel
from .combined import CombinedModel

architecture_func = {
    'hierarchical': HierarchicalModel,
    'vanilla': TransformerModel,
    'fact-aware': FactAwareModel,
    'mtl': TransformerModel,
    'combined': CombinedModel,
}

def get_module(name, base_arch=None):
    if name=='mtl' and base_arch:
        return architecture_func[base_arch]
    else:
        return architecture_func[name]