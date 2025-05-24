
import numpy as np
from genMoPlan.datasets.acrobot import _make_trajectory_indices
from genMoPlan.datasets.utils import compute_actual_length


actual_horizon_length = compute_actual_length(13, 10)
actual_history_length = compute_actual_length(3, 10)

use_history_padding = True
use_horizon_padding = False
stride = 10
arr = np.arange(200)



indices =_make_trajectory_indices(
    actual_history_length,
    actual_horizon_length,
    stride,
    use_history_padding,
    use_horizon_padding,
    True,
    np.ones((200, 10)),
    200,
    0
)

for i in indices:
    _, history_start, history_end, horizon_start, horizon_end, variation_type = i

    print(arr[history_start:history_end:stride], ' ', arr[horizon_start:horizon_end:stride])