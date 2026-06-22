try:
    from pyrometheus.bandit.impl.cantera import CanteraMechanism
except ImportError:
    CanteraMechanism = None

try:
    from pyrometheus.bandit.impl.plato import PlatoMechanism
except ImportError:
    PlatoMechanism = None
