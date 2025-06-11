
import numpy as np
from genMoPlan.datasets.utils import compute_actual_length


actual_horizon_length = compute_actual_length(13, 10)
actual_history_length = compute_actual_length(3, 10)

use_history_padding = True
use_horizon_padding = False
stride = 10
arr = np.arange(200)
