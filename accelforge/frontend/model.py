from accelforge.frontend.mapper.metrics import Metrics
from accelforge.util._basetypes import EvalableModel


class Model(EvalableModel):
    """Configuration for the model."""

    metrics: Metrics = Metrics.all_metrics()
    """
    Metrics to evaluate.

    If using spec to call mapper, leave this configuration as is. The mapper
    will make necessary configurations.
    """

    # _resource_usage_precision: float = 0
    # """
    # Rounds resource usage to the nearest multiple of this value. Must be between 0 and
    # 1. If zero, then no rounding is performed. NOTE: THE MULTI-PRECISION JOINING
    # OVERRIDES THIS. If you uncomment this, uncomment everything that accesses
    # something called _resource_usage_precision, and also update multi-precision
    # joining.
    # """

    # _objective_precision: float = 0
    # """
    # Rounds objective values to the nearest value representable by (1 + precision) ^ N.
    # Must be between 0 and 1. If zero, then no rounding is performed. NOTE: THE
    # MULTI-PRECISION JOINING OVERRIDES THIS. If you uncomment this, uncomment
    # everything that accesses something called _resource_usage_precision, and also
    # update multi-precision joining.
    # """
