import warnings
from torch import nn
from flow_matching.utils.manifolds import Manifold, Sphere, Product, FlatTorus, Euclidean
import torch
from enum import Enum, auto
import copy


class ManifoldType(Enum):
    SPHERE = auto()
    FLAT_TORUS = auto()
    PRODUCT = auto()
    EUCLIDEAN = auto()

class ManifoldWrapper:
    def __init__(self, manifold: Manifold):
        self._manifold = manifold
        self.manifold_type = self._determine_manifold_type(manifold)
        self.manifold_types = None
        
        # Pre-compute manifold types if using a product manifold
        if self.manifold_type == ManifoldType.PRODUCT:
            manifold_types = []

            for m in self._manifold.manifolds:
                manifold_types.append(self._determine_manifold_type(m))

                if manifold_types[-1] == ManifoldType.PRODUCT:
                    raise ValueError("Cannot nest product manifolds")

            self.manifold_types = manifold_types

        self.print_manifold_type()
    
    def __reduce__(self):
        """Implementation for pickling support."""
        if self.manifold_type == ManifoldType.PRODUCT:
            # For product manifolds, we need to capture each sub-manifold
            manifold_info = []
            for m, d in zip(self.manifold_types, self._manifold.dimensions):
                if m == ManifoldType.SPHERE:
                    manifold_info.append(('Sphere', d))
                elif m == ManifoldType.FLAT_TORUS:
                    manifold_info.append(('FlatTorus', d))
                elif m == ManifoldType.EUCLIDEAN:
                    manifold_info.append(('Euclidean', d))
                else:
                    raise ValueError(f"Unsupported manifold type for pickling: {m}")
            
            # Include all other attributes except _manifold
            state = {k: v for k, v in self.__dict__.items() if k != '_manifold'}
            
            # Return constructor and args
            return (self._reconstruct_wrapper, 
                    ('Product', manifold_info),  # What we need to reconstruct the manifold
                    state)                       # Other instance state
        else:
            # For non-product manifolds, just store their type and dimension
            if self.manifold_type == ManifoldType.SPHERE:
                manifold_info = ('Sphere', None)
            elif self.manifold_type == ManifoldType.FLAT_TORUS:
                manifold_info = ('FlatTorus', None)
            elif self.manifold_type == ManifoldType.EUCLIDEAN:
                manifold_info = ('Euclidean', None)
            else:
                raise ValueError(f"Unsupported manifold type for pickling: {self.manifold_type}")
            
            # Include all other attributes except _manifold
            state = {k: v for k, v in self.__dict__.items() if k != '_manifold'}
            
            # Return constructor and args
            return (self._reconstruct_wrapper, 
                    manifold_info,    # What we need to reconstruct the manifold
                    state)            # Other instance state
    
    @staticmethod
    def _reconstruct_wrapper(manifold_info, state):
        """Static method to reconstruct a ManifoldWrapper from pickle data."""
        # Product manifold case
        if manifold_info == 'Product':
            sub_manifolds = []
            total_dim = 0
            for sub_info in state:
                sub_type, sub_dim = sub_info
                if sub_type == 'Sphere':
                    sub_manifolds.append((Sphere(), sub_dim))
                elif sub_type == 'FlatTorus':
                    sub_manifolds.append((FlatTorus(), sub_dim))
                elif sub_type == 'Euclidean':
                    sub_manifolds.append((Euclidean(), sub_dim))
                else:
                    raise ValueError(f"Unknown manifold type: {sub_type}")
                total_dim += sub_dim
            manifold = Product(input_dim=total_dim, manifolds=sub_manifolds)
        # Single manifold case
        elif manifold_info == 'Sphere':
            manifold = Sphere()
        elif manifold_info == 'FlatTorus':
            manifold = FlatTorus()
        elif manifold_info == 'Euclidean':
            manifold = Euclidean()
        else:
            raise ValueError(f"Invalid manifold info: {manifold_info}")
        
        # Create the wrapper
        wrapper = ManifoldWrapper(manifold)
        
        # Restore the saved state
        wrapper.__dict__.update(state)
        
        return wrapper
    
    def compute_feature_dim(self, input_dim: int, n_fourier_features: int = None, manifold: Manifold = None):
        if manifold is None:
            manifold = self._manifold

        manifold_type = self._determine_manifold_type(manifold)

        if manifold_type == ManifoldType.SPHERE or manifold_type == ManifoldType.EUCLIDEAN:
            return input_dim
        elif manifold_type == ManifoldType.FLAT_TORUS:
            return input_dim * 2 * n_fourier_features
        elif manifold_type == ManifoldType.PRODUCT:
            return sum(self.compute_feature_dim(manifold.dimensions[i], n_fourier_features, m) for i, m in enumerate(manifold.manifolds))
        else:
            raise ValueError(f"Unsupported manifold type: {manifold_type}")

    def _determine_manifold_type(self, manifold):
        if isinstance(manifold, Sphere):
            return ManifoldType.SPHERE
        elif isinstance(manifold, FlatTorus):
            return ManifoldType.FLAT_TORUS
        elif isinstance(manifold, Product):
            return ManifoldType.PRODUCT
        elif isinstance(manifold, Euclidean):
            return ManifoldType.EUCLIDEAN
        else:
            raise ValueError(f"Unsupported manifold: {type(manifold)}")

    def __getattr__(self, name):
        # Avoid forwarding special method lookups that might lead to recursion
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")
        return getattr(self._manifold, name)
        
    def __deepcopy__(self, memo):
        # Custom deepcopy implementation to avoid recursion
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy all attributes that don't start with '_'
        for k, v in self.__dict__.items():
            if k != '_manifold':  # Handle _manifold separately
                setattr(result, k, copy.deepcopy(v, memo))
        
        # Copy the manifold
        setattr(result, '_manifold', copy.deepcopy(self._manifold, memo))
        
        return result

    def __dir__(self):
        return sorted(set(dir(self.__class__) + dir(self._manifold)))
   
    def print_manifold_type(self):
        if self.manifold_type == ManifoldType.PRODUCT:
            print(f"\n[ utils/manifold ] Manifold Type:")
            print(f"Manifold Product: [")
            for i, m in enumerate(self.manifold_types):
                print(f"  {m.name} x {self._manifold.dimensions[i]}")
            print("]\n")
        else:
            print(f"[ utils/manifold ] Manifold Type: {self.manifold_type.name}")
    
    def _sphere_wrap_params(self, x):
        batch_shape = x.shape[:-1]
        center = torch.zeros(*batch_shape, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        center[..., -1] = 1.0
        
        u = torch.zeros(*batch_shape, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        u[..., :-1] = x / 2
        
        return center, u
    
    def _zero_center_params(self, x):
        # For FlatTorus and Euclidean, we don't need to wrap the input
        return torch.zeros_like(x), x
    
    def _product_wrap_params(self, x):
        x_split = self._manifold.split(x)
        centers = []
        x_projs = []
        
        # Determine total output dimension
        total_center_dim = 0
        total_x_dim = 0
        
        for i, x_i in enumerate(x_split):
            if self.manifold_types[i] == ManifoldType.SPHERE:
                # Sphere adds an extra dimension
                total_center_dim += x_i.shape[-1] + 1
                total_x_dim += x_i.shape[-1] + 1
            else:
                # FlatTorus and Euclidean keep same dimensions
                total_center_dim += x_i.shape[-1]
                total_x_dim += x_i.shape[-1]
                
        # Process each component manifold
        current_center_idx = 0
        current_x_idx = 0
        
        # Pre-allocate output tensors if all batch dimensions match
        if all(x_i.shape[:-1] == x_split[0].shape[:-1] for x_i in x_split):
            batch_shape = x_split[0].shape[:-1]
            center_out = torch.zeros(*batch_shape, total_center_dim, device=x.device, dtype=x.dtype)
            x_out = torch.zeros(*batch_shape, total_x_dim, device=x.device, dtype=x.dtype)
            
            for i, x_i in enumerate(x_split):
                if self.manifold_types[i] == ManifoldType.SPHERE:
                    # Handle Sphere case
                    c_i, x_i_proj = self._sphere_wrap_params(x_i)
                    dim = c_i.shape[-1]
                    center_out[..., current_center_idx:current_center_idx+dim] = c_i
                    x_out[..., current_x_idx:current_x_idx+dim] = x_i_proj
                    current_center_idx += dim
                    current_x_idx += dim
                elif self.manifold_types[i] == ManifoldType.FLAT_TORUS or self.manifold_types[i] == ManifoldType.EUCLIDEAN:
                    # Handle FlatTorus case
                    dim = x_i.shape[-1]
                    # For FlatTorus, center is zeros and x is unchanged
                    # center_out zeros are already set from initialization
                    x_out[..., current_x_idx:current_x_idx+dim] = x_i
                    current_center_idx += dim
                    current_x_idx += dim
                else:
                    raise ValueError(f"Unsupported manifold type: {self.manifold_types[i]}")
            
            return center_out, x_out
        else:
            # Fall back to list-based approach if batch dimensions don't match
            for i, x_i in enumerate(x_split):
                if self.manifold_types[i] == ManifoldType.SPHERE:
                    center_i, x_i_proj = self._sphere_wrap_params(x_i)
                elif self.manifold_types[i] == ManifoldType.FLAT_TORUS or self.manifold_types[i] == ManifoldType.EUCLIDEAN:
                    center_i, x_i_proj = self._zero_center_params(x_i)
                else:
                    raise ValueError(f"Unsupported manifold type: {self.manifold_types[i]}")
                
                centers.append(center_i)
                x_projs.append(x_i_proj)
            
            return torch.cat(centers, dim=-1), torch.cat(x_projs, dim=-1)

    def wrap(self, x):
        if self.manifold_type == ManifoldType.SPHERE:
            center, u = self._sphere_wrap_params(x)
        elif self.manifold_type == ManifoldType.PRODUCT:
            center, u = self._product_wrap_params(x)
        elif self.manifold_type == ManifoldType.FLAT_TORUS:
            center, u = self._zero_center_params(x)
        elif self.manifold_type == ManifoldType.EUCLIDEAN:
            warnings.warn("Need not define Manifold for Euclidean space")
            return x, x
        else:
            raise ValueError(f"Unsupported manifold: {self.manifold_type}")
        
        return self._manifold.expmap(center, u)