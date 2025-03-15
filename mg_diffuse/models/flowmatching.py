from collections import namedtuple

from torch import nn, Tensor
import torch

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath, GeodesicProbPath
from flow_matching.solver import ODESolver, RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Manifold, Product, Euclidean


from .helpers import apply_conditioning
from .helpers.losses import Losses


Sample = namedtuple("Sample", "trajectories values chains")

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def wrap(manifold, samples):
    center = torch.zeros_like(samples)
    return manifold.expmap(center, samples)

class FourierFeatures(nn.Module):
    """Assumes input is in [0, 2pi]."""

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



class ProductFeatures(nn.Module):
    """The product of manifolds, Sphere, Torus and Euclidean."""
    """"TODO: Sphere_feature is id for now, update it"""

    def __init__(self, sphere_dim = 0, torus_dim = 0, euclidean_dim = 0):
        super().__init__()
        self.sphere_dim = sphere_dim
        self.torus_dim = torus_dim
        self.euclidean_dim = euclidean_dim
        self.size = [sphere_dim] + [1]*torus_dim + [euclidean_dim]

        self.torus_feature = FourierFeatures(1)

    def forward(self, x: Tensor) -> Tensor:
        x_splitted = torch.split(x, self.size, dim=-1)
        x_torus = [self.torus_feature(x_) for x_ in x_splitted[1:-1]] 
        x_torus = torch.cat(x_torus, dim=-1)

        # return = (sphere, torus, euclidean) = (x_splitted[0], x_torus,x_splitted[-1])
        return torch.cat((x_splitted[0], x_torus, x_splitted[-1]), dim=-1)


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, model: nn.Module, manifold: Manifold):
        super().__init__()
        self.model_euclidean = model  # nn model has an Euclidean input
        self.manifold = manifold

        # adding layer Manifold -> Euclidean
        self.input_layer = ProductFeatures(manifold.sphere_dim, manifold.torus_dim, manifold.euclidean_dim)

        features_dim = 2*manifold.sphere_dim + 2*manifold.torus_dim + manifold.euclidean_dim
        output_dim = manifold.sphere_dim + manifold.torus_dim + manifold.euclidean_dim

        # adding extra layer to the model: Euclidean features_dim -> Euclidean output_dim
        self.output_layer = nn.Linear(features_dim, output_dim)  # reducing the dimension here to not modify the original self.model_euclidean

        

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.manifold.projx(x)
        x_in = self.input_layer(x)
        v = self.model_euclidean(x_in, t)
        v = self.output_layer(v)
        v = self.manifold.proju(x, v)
        return v
    


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        t=t.unsqueeze(0)  # for the model t has to have dimension but the ODE solver uses 0-D t
        return self.model(x, t)


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    # t_int = (self.n_timesteps*t).long()
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class FlowMatching(nn.Module):
    def __init__(
        self,
        model,
        observation_dim,
        horizon,
        n_timesteps=100,
        clip_denoised=False,
        predict_epsilon=True,
        loss_type="l1",
        loss_weights=None,
        loss_discount=1.0,
        manifold=Euclidean()
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.transition_dim = observation_dim
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.n_timesteps = int(n_timesteps)
        self.horizon = horizon

        self.device = DEVICE

        loss_weights = self.get_loss_weights(loss_discount, loss_weights)

        self.loss_fn = Losses[loss_type](loss_weights)

        self.manifold = manifold
        if manifold==Euclidean() or (self.manifold.sphere_dim == 0 and self.manifold.torus_dim == 0):
            self.manifold = Euclidean()  # overide Product (product of manifolds) since Euclidean() is simpler
            self.model = model  # velocity field model init
            self.prob_path = AffineProbPath(scheduler=CondOTScheduler())
            wrapped_vf = WrappedModel(self.model)  # wrapped vector field model to be used for ODE solver
            self.solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class

        else:
            self.model = ProjectToTangent(model, manifold=manifold)  # velocity field: Ensures we can just use Euclidean divergence.
            self.prob_path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold) # instantiate an geodesic path object
            wrapped_vf = WrappedModel(self.model)  # wrapped vector field model to be used for ODE solver
            self.solver = RiemannianODESolver(velocity_model=wrapped_vf, manifold=manifold)  # create an ODESolver class on Manifold

    def get_loss_weights(self, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        return loss_weights

    @torch.no_grad()
    def flow(
        self,
        shape,
        cond,
        verbose=True,
        return_chain=False,
        **sample_kwargs
    ):
        """
        Apply conditioning to x by fixing the states in x at the given timesteps from cond
        Then solve ODE to get the flow solution, the path is constant at the condition
        """
        # step size for ode solver
        step_size = 0.05
        T = torch.linspace(0,1,self.n_timesteps, device=self.device)  # sample times 
        # T = T.to(device=device)

        # x_init = torch.randn(shape, device=device)
        x_init = torch.randn(shape, device=self.device)
        x_init = apply_conditioning(x_init, cond)
        x_init = wrap(self.manifold, x_init)

        # solve ode
        sol = self.solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  

        x_end = sol[-1]  # the end point of the flow (generated distribution)

        chain = self.manifold, chain if return_chain else None  # from the normal distribution to last step (t=1, the predition)

        # x_end = wrap(self.manifold, x_end)
        # chain = wrap(self.manifold, chain) if return_chain else None  # from the normal distribution to last step (t=1, the predition)

        values = torch.zeros(len(x_end), device=x_end.device)  # values for future implementation
        x_end, values = sort_by_values(x_end, values)  # identity for now



        return Sample(x_end, values, chain)


    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        """
        conditions : [ (time, state), ... ]
        """
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.flow(shape, cond, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#
    
    def loss_vf(self, x, cond):
        # Get a normal distribution of noise
        # Apply conditioning to the noise
        noise = torch.randn_like(x)
        x_noisy = apply_conditioning(noise, cond)

        # Apply conditioning to the target x
        x_target = apply_conditioning(x, cond)

        assert noise.shape == x_target.shape
        x_noisy = wrap(self.manifold, x_noisy)

        batch_size = len(x)
        t = torch.rand((batch_size,), device=x.device)


        path_sample = self.prob_path.sample(t=t, x_0=x_noisy, x_1=x_target)

        return self.loss_fn(self.model(path_sample.x_t, path_sample.t), path_sample.dx_t)
    
    def loss(self, x, *args):
        """
        Calculate the loss for the model vector field
        """
        return self.loss_vf(x, *args)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
