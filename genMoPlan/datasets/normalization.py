import numpy as np
import warnings

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, params: dict = None, **kwargs) -> None: # type: ignore
        if params is None:
            raise ValueError('params must be provided')
        
        self.params = params

        for key, value in params.items():
            setattr(self, key, value)

    def __call__(self, x) -> np.ndarray:
        return self.normalize(x)

    def normalize(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    '''
        identity function
    '''

    def normalize(self, *args, **kwargs) -> np.ndarray:
        x: np.ndarray = args[0]
        return x

    def unnormalize(self, *args, **kwargs) -> np.ndarray:
        x: np.ndarray = args[0]
        return x


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''
    means: np.ndarray = None # type: ignore
    stds: np.ndarray = None # type: ignore
    X: np.ndarray = None # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = args[0]

        self._indices_to_normalize = []

        # Check if params in kwargs
        if self.means is None or self.stds is None:
            print(" [datasets/normalization] Means and stds not provided, computing from data")
        
            self.means = self.X.mean(axis=(0, 1))
            self.stds = self.X.std(axis=(0, 1))

            self.params['means'] = list(self.means)
            self.params['stds'] = list(self.stds)

            self._indices_to_normalize = list(range(len(self.means)))
        else:
            means = []
            stds = []
            for i in range(len(self.params['means'])):
                if self.params['means'][i] is not None:
                    means.append(self.params['means'][i])
                    stds.append(self.params['stds'][i])
                    self._indices_to_normalize.append(i)

            self.means = np.array(means)
            self.stds = np.array(stds)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.means.size}\n    '''
            f'''means: {np.round(self.params['means'], 2)}\n    '''
            f'''stds: {np.round(self.params['stds'], 2)}\n'''
        )

    def normalize(self, *args, **kwargs) -> np.ndarray:
        x: np.ndarray = args[0].copy()
        x[..., self._indices_to_normalize] = (x[..., self._indices_to_normalize] - self.means) / self.stds
        return x

    def unnormalize(self, *args, **kwargs) -> np.ndarray:
        x: np.ndarray = args[0].copy()
        x[..., self._indices_to_normalize] = x[..., self._indices_to_normalize] * self.stds + self.means
        return x


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''
    mins: np.ndarray = None # type: ignore
    maxs: np.ndarray = None # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._indices_to_normalize = []
        
        if self.mins is None or self.maxs is None:
            raise ValueError('mins and maxs must be provided')
        else:
            mins = []
            maxs = []
            for i in range(len(self.params['mins'])):
                if self.params['mins'][i] is not None:
                    mins.append(self.params['mins'][i])
                    maxs.append(self.params['maxs'][i])
                    self._indices_to_normalize.append(i)
            self.mins = np.array(mins)
            self.maxs = np.array(maxs)

    def __repr__(self) -> str:
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def normalize(self, *args, **kwargs) -> np.ndarray:
        x: np.ndarray = args[0]
        x = x.copy()
        eps = kwargs.get('eps', 1e-6)

        
        
        if not np.all(x[..., self._indices_to_normalize].min(axis=0) >= self.mins - eps):
            warnings.warn(f'Min value {x[..., self._indices_to_normalize].min(axis=0)} < Limit Min Value {self.mins} (tolerance: {eps})')
        if not np.all(x[..., self._indices_to_normalize].max(axis=0) <= self.maxs + eps):
            warnings.warn(f'Max value {x[..., self._indices_to_normalize].max(axis=0)} > Limit Max Value {self.maxs} (tolerance: {eps})')

        ## [ 0, 1 ]
        x[..., self._indices_to_normalize] = (x[..., self._indices_to_normalize] - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x[..., self._indices_to_normalize] = 2 * x[..., self._indices_to_normalize] - 1

        if not np.all(x[..., self._indices_to_normalize].min(axis=0) >= -1 - eps):
            warnings.warn(f'Normalized Min value {x[..., self._indices_to_normalize].min(axis=0)} < -1 (tolerance: {eps})')
        if not np.all(x[..., self._indices_to_normalize].max(axis=0) <= 1 + eps):
            warnings.warn(f'Normalized Max value {x[..., self._indices_to_normalize].max(axis=0)} > 1 (tolerance: {eps})')

        return x

    def unnormalize(self, *args, **kwargs) -> np.ndarray:
        x: np.ndarray = args[0].copy()
        eps: float = kwargs.get('eps', 1e-4)
        '''
            x : [ -1, 1 ]
        '''
        if x[..., self._indices_to_normalize].max() > 1 + eps or x[..., self._indices_to_normalize].min() < -1 - eps: 
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x[..., self._indices_to_normalize] = np.clip(x[..., self._indices_to_normalize], -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x[..., self._indices_to_normalize] = (x[..., self._indices_to_normalize] + 1) / 2.

        x[..., self._indices_to_normalize] = x[..., self._indices_to_normalize] * (self.maxs - self.mins) + self.mins

        return x
    
def get_normalizer(normalizer_name: str, params: dict) -> Normalizer:
    normalizer_class = eval(normalizer_name)
    return normalizer_class(params=params)