import torch
from torch import nn, Tensor
from genMoPlan.utils.manifold import ManifoldWrapper, ManifoldType

class FourierFeatures(nn.Module):
    def __init__(self, n_fourier_features: int):
        super().__init__()
        self.n_fourier_features = n_fourier_features

    def forward(self, x: Tensor) -> Tensor:
        feature_vector = [
            torch.sin((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        feature_vector += [
            torch.cos((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        return torch.cat(feature_vector, dim=-1)

class IdentityFeatureLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x, query):
        return x
    

class FlatTorusFeatureLayer(nn.Module):
    def __init__(self, input_dim: int, n_fourier_features: int = 1):
        super().__init__()
        self.input_dim = input_dim

        self.layer = FourierFeatures(n_fourier_features)

    def forward(self, x, query):
        return self.layer(x)


class ProductFeatureLayer(nn.Module):
    def __init__(self, manifold: ManifoldWrapper, input_dim: int, n_fourier_features: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.manifold = manifold
        input_layers = []

        for i, manifold_type in enumerate(manifold.manifold_types):
            if manifold_type == ManifoldType.SPHERE or manifold_type == ManifoldType.EUCLIDEAN:
                input_layers.append(IdentityFeatureLayer(manifold.dimensions[i]))

            elif manifold_type == ManifoldType.FLAT_TORUS:
                input_layers.append(FlatTorusFeatureLayer(manifold.dimensions[i], n_fourier_features))

            else:
                raise ValueError(f"Unsupported manifold type: {manifold_type}")
            
        self.input_layers = nn.ModuleList(input_layers)

    def forward(self, x, query):
        x_split = self.manifold.split(x)
        if query is not None:
            query_split = self.manifold.split(query)
            return torch.cat([layer(x_split[i], query_split[i]) for i, layer in enumerate(self.input_layers)], dim=-1)
        else:
            return torch.cat([layer(x_split[i], query) for i, layer in enumerate(self.input_layers)], dim=-1)


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, model: nn.Module, manifold: ManifoldWrapper, input_dim: int, n_fourier_features: int = 1):
        super().__init__()
        self.model = model
        self.manifold = manifold

        if self.manifold.manifold_type == ManifoldType.SPHERE:
            self.manifold_features_layer = IdentityFeatureLayer(manifold, input_dim)
        elif self.manifold.manifold_type == ManifoldType.FLAT_TORUS:
            self.manifold_features_layer = FlatTorusFeatureLayer(manifold, input_dim, n_fourier_features)
        elif self.manifold.manifold_type == ManifoldType.PRODUCT:
            self.manifold_features_layer = ProductFeatureLayer(manifold, input_dim, n_fourier_features)
        else:
            raise ValueError(f"Unsupported manifold type: {self.manifold.manifold_type}")
        
    def forward(self, x, query, t):
        x = self.manifold.projx(x)
        manifold_features = self.manifold_features_layer(x, query)
        v = self.model(manifold_features, query, t)
        v = self.manifold.proju(x, v)
        return v
