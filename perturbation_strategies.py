import os
import sys
import numpy as np

np.random.seed(42)
import scipy.stats
from abc import ABC, abstractmethod

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from pytftk.arrays import set_value, get_value


class PerturbationStrategy(ABC):
    @abstractmethod
    def perturb(self, sequence, axis, dim, intensity, **kwargs):
        """Perturb the input sequence on the given axis and intensity.
        Args:
            sequence: input sequence, shape (timesteps, nodes, features)
            axis: axis to perturb, possible values: "timesteps", "nodes", "features"
            dim: dimension to perturb, possible values range from 0 to the length of the axis. If None, perturb all the dimensions.
            intensity: intensity of the perturbation, a value from 0.0 to 1.0
        Returns:
            perturbed_sequence: return a sequence of shape (timesteps, nodes, features)
            where the dim-th element of the specified axis is perturbed by the given intensity.
        """
        if sequence.ndim != 3:
            raise ValueError(
                f"Wrong shape for the input sequence: {sequence.shape}. Expected: (timesteps, nodes, features)"
            )
        if axis not in ["timesteps", "nodes", "features"]:
            raise ValueError(
                f"Wrong axis: {axis}. Possible values: 'timesteps', 'nodes', 'features'"
            )
        self._axis = {"timesteps": 0, "nodes": 1, "features": 2}[axis]


class PlusMinusSigmaPerturbationStrategy(PerturbationStrategy):
    def perturb(self, sequence, axis, dim, intensity, **kwargs):
        super().perturb(sequence, axis, dim, intensity, **kwargs)

        if dim is None:
            dim = range(sequence.shape[self._axis])
        for d in dim:
            std = np.std(get_value(sequence, self._axis, d))
            if kwargs.get("type") == "percentile":
                std = scipy.stats.norm.ppf(intensity, loc=0, scale=std)
                print("applying percentile")
            if kwargs.get("minus", False):
                std *= -1
            print(d, std)
            sequence = set_value(sequence, self._axis, d, std, update=True)

        return sequence


class NormalPerturbationStrategy(PerturbationStrategy):
    def perturb(self, sequence, axis, dim, intensity, **kwargs):
        """Perturb the input sequence on the given axis and dimension(s), replacing
         them with values sampled from a normal distribution, having the original
          value as mean and the standard deviation according to the "type" kwarg.
        Args:
            sequence: input sequence, shape (timesteps, nodes, features)
            axis: axis to perturb, possible values: "timesteps", "nodes", "features"
            dim: dimension to perturb, possible values range from 0 to the length of the axis. If None, perturb all the dimensions.
            intensity: intensity of the perturbation, a value from 0.0 to 1.0
            type: type of perturbation, possible values: "stddev", "relative", "absolute":
            - if "stddev", the standard deviation of the normal distribution is the standard deviation of the slice multiplied by the intensity;
            - if "relative", the standard deviation of the normal distribution is the mean of the slice multiplied by the intensity;
            - if "absolute", the standard deviation of the normal distribution is the intensity.
        Returns:
            perturbed_sequence: return a sequence of shape (timesteps, nodes, features)
            where the dim-th element of the specified axis is perturbed by the given intensity.
        Notes:
            The "stddev" type produces higher perturbations for more skewed features,
            The "relative" type produces higher perturbations for features that
            have generally a high value, even if constant
            The "absolute" type, produces equal perturbations, but they may affect
            features having a generally lower value more heavily
        """
        super().perturb(sequence, axis, dim, intensity, **kwargs)

        if kwargs.get("type") not in [None, "stddev", "relative", "absolute"]:
            raise ValueError(
                "kwarg 'type' can only be one of ['stddev', 'relative', 'absolute']"
            )

        perturbation_type = kwargs.get("type", "stddev")
        slice = get_value(sequence, self._axis, dim)
        std = (
            np.std(slice) * intensity
            if perturbation_type == "stddev"
            else (
                np.mean(slice) * intensity
                if perturbation_type == "relative"
                else intensity
            )
        )
        return set_value(
            sequence,
            self._axis,
            dim,
            np.random.normal(0.0, std, size=slice.shape),
            update=True,
        )


class PercentilePerturbationStrategy(PerturbationStrategy):
    def perturb(self, sequence, axis, dim, intensity, **kwargs):
        """_summary_

        Args:
            sequence (_type_): _description_
            axis (_type_): _description_
            dim (_type_): _description_
            intensity (float): the percentile
        """
        super().perturb(sequence, axis, dim, intensity, **kwargs)

        pert = scipy.stats.norm.ppf(intensity, loc=0, scale=1)
        if np.random.rand() < 0.5:
            pert *= 1

        return set_value(
            sequence,
            self._axis,
            dim,
            pert,
            update=True,
        )


class FixedValuePerturbationStrategy(PerturbationStrategy):
    def perturb(self, sequence, axis, dim, intensity, **kwargs):
        """Perturb the input sequence on the given axis and dimension(s), replacing
         them with a fixed value.
         Args:
            sequence: input sequence, shape (timesteps, nodes, features)
            axis: axis to perturb, possible values: "timesteps", "nodes", "features"
            dim: dimension(s) to perturb, can be an integer ranging from 0 to the length of the axis or a list of integers
            intensity: value to replace the original value with
            update: if True, add the perturbation to the original value, otherwise replace it
        Returns:
            perturbed_sequence: return a sequence of shape (timesteps, nodes, features)
            where the dim-th element of the specified axis is perturbed by the given intensity.
        """
        super().perturb(sequence, axis, dim, intensity, **kwargs)

        return set_value(
            sequence, self._axis, dim, intensity, update=kwargs.get("update")
        )
