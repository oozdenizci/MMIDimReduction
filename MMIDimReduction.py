#!/usr/bin/env python
import numpy as np
import chainer
from chainer import Variable, optimizers
import chainer.functions as F
import chainer.links as L
from chainer.initializers import LeCunUniform


class MMINet:
    """ This object obtains a non-parametric linear/nonlinear feature transformation based on 
        a stochastic estimate of the mutual information of projected features and class labels. 
        The method uses the Stochastic Mutual Information Gradient (SMIG) based on non-parametric 
        entropy estimates to update the projection network weights.
        METHODS:
            learn - obtains the feature transformation network via the information theoretic loss
            reduce - reduces dimensionality of features using the learned network
    """
    def __init__(self, input_dim, output_dim=2, net='linear'):
        """ Initialize object
            VARIABLES:
                input_dim - input dimensionality of the features provided to the network
                output_dim - output dimensionality of the feature projection network (default: 2)
                net - network architecture to be used (i.e., "linear" (default) or "nonlinear")
        """
        if net == 'linear':
            self.f = Linear(input_dim, output_dim)
        if net == 'nonlinear':
            self.f = MLP(input_dim, output_dim)
        self.optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        self.optimizer.setup(self.f)

    def learn(self, input_array, input_labels, num_epochs=5):
        """ Learn the transformation network using the information theoretic loss
            INPUT:
                input_array - input feature array to be used for learning [num_trials x input_dim]
                input_labels - input feature labels array [num_trials x 1]
        """
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            trial_indices = np.arange(input_labels.size)
            np.random.shuffle(trial_indices)
            for ind in trial_indices:
                self.f.cleargrads()
                proj_array = self.f(Variable(input_array))
                loss = self._stochasticMI(proj_array, input_labels, ind)
                loss.backward()
                self.optimizer.update()

    def _stochasticMI(self, proj_array, input_labels, ind):
        """ Instantaneous MI estimate between projected features and labels based on non-parametric density estimates
            INPUT:
                proj_array - Variable of projected features array [num_trials x output_dim]
                input_labels - input feature labels array [num_trials x 1]
                ind - index of the instantaneous feature for mutual information estimation
            OUTPUT:
                Variable of negated instantaneous mutual information
        """
        # Empirical class prior estimates
        num_classes = np.max(input_labels) + 1
        num_samples = len(input_labels) - 1
        obs_labels = [np.where(np.delete(input_labels, ind) == c)[0] for c in range(num_classes)]
        priors = [len(obs_labels[c]) / num_samples for c in range(num_classes)]

        # Class conditional kernel density estimate value components
        constants, energies, lse_energies = [], [], []
        for c in range(num_classes):
            const, energy = self._kdeparts(proj_array[obs_labels[c]], proj_array[ind])
            constants.append(const)
            energies.append(energy)
            lse_energies.append(F.logsumexp(energy).data)

        # Use the maximum logsumexp(energy) across classes for the exp-normalize trick
        max_index = lse_energies.index(max(lse_energies))
        joint_prob = [priors[c] * constants[c] * F.exp(F.logsumexp(energies[c]) - F.logsumexp(energies[max_index])) for c in range(num_classes)]

        # Calculate entropy and conditional entropy for the stochastic MI estimate
        conditional_entropy_parts = []
        entropy_parts = []
        for c in range(num_classes):
            c_given_y = joint_prob[c] / sum(joint_prob)
            conditional_entropy_parts.append(c_given_y * (F.log(constants[c]) + F.logsumexp(energies[c])))
            entropy_parts.append(priors[c] * constants[c] * F.exp(F.logsumexp(energies[c])))
        conditional_entropy = sum(conditional_entropy_parts)
        entropy = F.log(sum(entropy_parts))

        return entropy - conditional_entropy

    def _kdeparts(self, input_obs, input_ins):
        """ Multivariate Kernel Density Estimation (KDE) with Gaussian kernels on the given random variables.
            INPUT:
                input_obs - Variable of input observation random variables to estimate density
                input_ins - Variable of input data instance to calculate the probability value
            OUTPUT:
                const - Constant term in the Gaussian KDE expression
                energy - Expressions in the exponential to calculate Gaussian KDE (energy wrt. every obs. point)
        """
        [n, d] = input_obs.shape

        # Compute Kernel Bandwidth Matrix based on Silverman's Rule of Thumb
        silverman_factor = np.power(n * (d + 2.0) / 4.0, -1. / (d + 4))
        input_centered = input_obs - F.mean(input_obs, axis=0, keepdims=True)
        data_covariance = F.matmul(F.transpose(input_centered), input_centered) / n
        kernel_bw = F.diagonal(data_covariance) * (silverman_factor ** 2) * np.eye(d, d)
        const = 1 / (n * ((2 * np.pi) ** (d/2)) * F.sqrt(F.det(kernel_bw)))

        # Compute energy expressions in the exponent for every observation point
        diff = input_obs - input_ins
        energy = -0.5 * F.diagonal(F.matmul(F.matmul(diff, F.inv(kernel_bw)), F.transpose(diff)))

        return const, energy

    def reduce(self, input_array):
        """ Reduce dimensionality of given array using the learned network
            INPUT:
                input_array - input feature array to be used for learning [num_trials x input_dim]
            OUTPUT:
                output_array - output feature array after transformation [num_trials x output_dim]
        """
        return self.f(Variable(input_array)).data


class MLP(chainer.Chain):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(MLP, self).__init__(
            l1=L.Linear(input_dim, int(input_dim//2), initialW=LeCunUniform(), nobias=True),
            l2=L.Linear(int(input_dim//2), int(input_dim//4), initialW=LeCunUniform(), nobias=True),
            l3=L.Linear(int(input_dim//4), output_dim, initialW=LeCunUniform(), nobias=True),
        )

    def __call__(self, x):
        return self.l3(F.elu(self.l2(F.elu(self.l1(x)))))


class Linear(chainer.Chain):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(Linear, self).__init__(
            l1=L.Linear(input_dim, output_dim, initialW=LeCunUniform(), nobias=True),
        )

    def __call__(self, x):
        return self.l1(x)
