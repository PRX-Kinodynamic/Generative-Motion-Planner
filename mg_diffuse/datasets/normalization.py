import numpy as np
import scipy.interpolate as interpolate
import torch

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer):

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizer = normalizer(dataset)

    def __repr__(self):
        return str(self.normalizer)

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X=None, params=None):
        if X is None and params is None:
            raise ValueError('Either dataset or params must be provided')

        if params is not None:
            self.params = params
            self.mins = np.array(params['mins'], np.float32)
            self.maxs = np.array(params['maxs'], np.float32)
            self.X = None
        else:
            self.mins = X.min(axis=(0, 1))
            self.maxs = X.max(axis=(0, 1))
            self.X = X.astype(np.float32)
            self.params = {
                'mins': list(self.mins),
                'maxs': list(self.maxs)
            }

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    '''
        identity function
    '''

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if params in kwargs
        if 'params' in kwargs:
            self.means = np.array(kwargs['params']['means'], np.float32)
            self.stds = np.array(kwargs['params']['stds'], np.float32)
        else:
            self.means = self.X.mean(axis=(0, 1))
            self.stds = self.X.std(axis=(0, 1))

        self.params['means'] = list(self.means)
        self.params['stds'] = list(self.stds)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but can handle data for which a dimension is constant
    '''

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        if 'params' in kwargs:
            self.mins = np.array(kwargs['params']['mins'], np.float32)
            self.maxs = np.array(kwargs['params']['maxs'], np.float32)
        else:
            for i in range(len(self.mins)):
                if self.mins[i] == self.maxs[i]:
                    print(f'''
                        [ utils/normalization ] Constant data in dimension {i} | '''
                        f'''max = min = {self.maxs[i]}'''
                    )
                    self.mins -= eps
                    self.maxs += eps

        self.params['mins'] = list(self.mins)
        self.params['maxs'] = list(self.maxs)

#-----------------------------------------------------------------------------#
#------------------------------- CDF normalizer ------------------------------#
#-----------------------------------------------------------------------------#

class CDFNormalizer(Normalizer):
    '''
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)
        ]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        ## reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

class WrapManifold(Normalizer):
    '''
        maps [ x_euclidean_min, x_euclidean_max ] to [ -1, 1 ] and 
        wraps the coordinates on non-euclidean manifold 
    '''
    def __init__(self, manifold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        self.maxs = self.maxs[manifold.sphere_dim + manifold.torus_dim:]
        self.mins = self.mins[manifold.sphere_dim + manifold.torus_dim:]

        assert self.mins.shape[-1] == manifold.euclidean_dim


    def wrap(self, x_sphere, x_torus, x_euclidean):
        """ wrap the coordinates on non-euclidean manifold"""

        center_euclidean = torch.zeros_like(x_euclidean) #euclidean
        center_torus = torch.zeros_like(x_torus) #torus
        
        #sphere
        center_sphere = torch.cat([torch.zeros_like(x_sphere), torch.ones_like(x_sphere[..., 0:1])], dim=-1)
        x_sphere = torch.cat([x_sphere, torch.zeros_like(x_sphere[..., 0:1])], dim=-1) / 2

        # concatenate
        center = torch.cat((center_sphere, center_torus, center_euclidean), dim=-1)
        x = torch.cat((x_sphere, x_torus, x_euclidean), dim=-1)

        return self.manifold.expmap(center, x).detach().cpu().numpy()


    def normalize(self, x):
        '''
            apply wrap for non-euclidean coordinates and 
            normalize (x_euclidean to [-1,1] ) for euclidean coordinates
        '''
        x = torch.from_numpy(x)
        x_sphere, x_torus, x_euclidean = self.manifold.split(x)

        # euclidean
        ## [ 0, 1 ]
        x_euclidean = (x_euclidean - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x_euclidean = 2 * x_euclidean - 1

        return self.wrap(x_sphere, x_torus, x_euclidean)

    def unnormalize(self, x, eps=1e-4):
        '''
            apply wrap for non-euclidean coordinates and 
            unnormalize ([-1,1] to x_euclidean) for euclidean coordinates
        '''
        x = torch.from_numpy(x)
        x_sphere, x_torus, x_euclidean = self.manifold.split(x)

        # euclidean
        if x_euclidean.max() > 1 + eps or x_euclidean.min() < -1 - eps:

            x_euclidean = torch.clip(x_euclidean, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x_euclidean = (x_euclidean + 1) / 2.
        x_euclidean = x_euclidean * (self.maxs - self.mins) + self.mins

        return self.wrap(x_sphere, x_torus, x_euclidean)
