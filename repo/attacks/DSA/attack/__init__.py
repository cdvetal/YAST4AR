from .i_fgsm import FGSM, IFGSM
from .mi_fgsm import MIFGSM
from .dsa import DSA


def _get_cfg_defaults():
    """Best-effort access to the original package defaults.

    The upstream DSA package uses YACS config objects. In this thesis repo we
    often do not want to depend on those configs (or YACS) at runtime.
    """
    try:
        from config.attack_config import cfg

        return {
            'steps': cfg.attack.steps,
            'epsilon': cfg.attack.eps,
            'budget': cfg.attack.budget,
            'num_classes': cfg.dataset.num_classes,
        }
    except Exception:
        return {
            'steps': int(1e4),
            'epsilon': 16 / 255,
            'budget': 4000,
            'num_classes': None,
        }


def get_attack(
    name,
    *,
    epsilon=None,
    steps=None,
    budget=None,
    local_models=None,
    num_classes=None,
    **kwargs,
):
    defaults = _get_cfg_defaults()
    if epsilon is None:
        epsilon = defaults['epsilon']
    if steps is None:
        steps = defaults['steps']
    if budget is None:
        budget = defaults['budget']
    if num_classes is None:
        num_classes = defaults['num_classes']

    attack = eval(
        name
        + "(local_models=local_models, budget=budget, epsilon=epsilon, num_classes=num_classes, **kwargs)"
    )
    return attack
