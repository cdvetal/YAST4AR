import sys

# Ensure package can be imported using either 'autoattack' or 'AutoAttack'
# Some files use absolute imports like 'from autoattack import ...' or 'from AutoAttack import ...'
# Map those top-level names to this package module so imports resolve when this
# folder is not installed as a top-level package. We register the mapping before
# importing submodules so their absolute imports (e.g., `from autoattack import checks`)
# can find this package.
sys.modules['autoattack'] = sys.modules.get(__name__)
sys.modules['AutoAttack'] = sys.modules.get(__name__)

from .autoattack import AutoAttack
