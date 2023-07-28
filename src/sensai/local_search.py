import collections
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Type, Sequence, TypeVar, Generic

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .util.aggregation import RelativeFrequencyCounter

log = logging.getLogger(__name__)


class SATemperatureSchedule(ABC):
    """
    Describes how temperature changes as the annealing process goes on.
    The function maps a degree of completion (in the interval from 0 to 1) to a temperature value.
    """

    @abstractmethod
    def temperature(self, degree_of_completion: float) -> float:
        """
        Computes the temperature for a degree of completion in [0, 1]
        :param degree_of_completion: the degree to which the simulated annealing process has completed in [0, 1]
        :return: the temperature
        """
        pass

    def probability(self, degree_of_completion: float, cost_delta: float) -> Tuple[float, float]:
        T = self.temperature(degree_of_completion)
        p = math.exp(-cost_delta / T) if T > 0.0 else 0.0
        return p, T

    @abstractmethod
    def _get_params(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}{self._get_params()}"


class SATemperatureScheduleFixed(SATemperatureSchedule):
    """A schedule with a constant temperature"""
    def __init__(self, t):
        self.t = t

    def temperature(self, degree_of_completion):
        return self.t

    def _get_params(self):
        return {"fixedTemp": self.t}


class SATemperatureScheduleExponential(SATemperatureSchedule):
    """ A temperature schedule for simulated annealing where the temperature drops exponentially. """

    def __init__(self, t0, t1, exponent_factor):
        """
        :param t0: temperature at the beginning (degree of completion 0)
        :param t1: temperature at the end (degree of completion 1), which must be smaller than t0 and larger than 0
        :param exponent_factor: factor with which to multiply the exponent (the larger the factor, the faster the temperature drops)
        """
        super().__init__()
        if t1 > t0 or t0 <= 0:
            raise ValueError("Inadmissible temperatures given.")

        if exponent_factor < 1:
            raise ValueError("The exponent factor cannot be less than 1.")

        self.t0 = t0
        self.t1 = t1
        self.t_drop = t0 - t1
        self.exponent_factor = exponent_factor

    def temperature(self, degree_of_completion):
        return self.t0 - self.t_drop * (1.0 - math.exp(-self.exponent_factor * degree_of_completion))

    def _get_params(self):
        return {"t0": self.t0, "t1": self.t1, "exp_factor": self.exponent_factor}


def reverse_sigmoid(degree_of_completion, v0, v1, steepness, mid_degree=0.5):
    return v0 - (v0-v1) / (1.0 + math.exp(-steepness * (degree_of_completion - mid_degree)))


class SATemperatureScheduleReverseSigmoid(SATemperatureSchedule):
    """
    A temperature schedule for simulated annealing where the temperature drops in a reverse sigmoid curve (based on the logistic function)
    """

    def __init__(self, t0, t1, steepness, mid_degree=0.5):
        """
        :param t0: temperature at the beginning (degree of completion 0)
        :param t1: temperature at the end (degree of completion 1), which must be smaller than t0 and larger than 0
        :param steepness: Factor which roughly corresponds to the (negative) slope at the inflection point 0.5.
            For the sigmoid shape to be sufficiently pronounced, a value of at least 5 is recommended (the closer the value is to 1,
            the more the curve approaches a linear decay).
        :param mid_degree: the degree of completion at which the function shall return the temperature (t0+t1)/2
        """
        super().__init__()
        if t1 > t0 or t0 <= 0:
            raise ValueError("Inadmissible temperatures given.")
        if mid_degree < 0 or mid_degree > 1:
            raise Exception("Mid-degree must be between 0 and 1")

        self.t0 = t0
        self.t1 = t1
        self.steepness = steepness
        self.mid_degree = mid_degree

    def temperature(self, degree_of_completion):
        return reverse_sigmoid(degree_of_completion, self.t0, self.t1, self.steepness, self.mid_degree)

    def _get_params(self):
        return {"t0": self.t0, "t1": self.t1, "steepness": self.steepness, "mid_degree": self.mid_degree}


class SATemperatureScheduleReverseSigmoidSymmetric(SATemperatureScheduleReverseSigmoid):
    """
    A variant of the logistic schedule with a reverse sigmoid shape, where the probability of acceptance for a given assumed positive
    cost delta is symmetric. "Symmetric" here means that half the schedule is to be above 0.5 and half of it below 0.5.
    """
    def __init__(self, t0, t1, steepness, cost_delta_for_symmetry):
        """
        :param t0: temperature at the beginning (degree of completion 0)
        :param t1: temperature at the end (degree of completion 1), which must be smaller than t0 and larger than 0
        :param steepness: Factor which roughly corresponds to the (negative) slope at the inflection point 0.5.
            For the sigmoid shape to be sufficiently pronounced, a value of at least 8 is recommended (the closer the value is to 1,
            the more the curve approaches a linear decay).
        :param cost_delta_for_symmetry: the (positive) cost delta value which the curve shall be "symmetric"
        """
        k = steepness
        cdelta = cost_delta_for_symmetry
        mid_degree = -(2*math.log(-(13614799*t0-19642003*cdelta)/(13614799*t1-19642003*cdelta))-k)/(2*k)
        super().__init__(t0, t1, steepness, mid_degree)


class SATemperatureSchedulePower(SATemperatureSchedule):
    """
    A temperature schedule for simulated annealing where the temperature drops with powers of the degree of completion d,
    i.e. the temperature drop is proportional to d to the power of e for a given exponent e.
    """

    def __init__(self, t0, t1, exponent):
        """
        :param t0: temperature at the beginning (degree of completion 0)
        :param t1: temperature at the end (degree of completion 1), which must be smaller than t0 and larger than 0
        :param exponent: the exponent of the power function
        """
        super().__init__()
        if t1 > t0 or t0 <= 0:
            raise ValueError("Inadmissible temperatures given.")

        self.t0 = t0
        self.t1 = t1
        self.tDrop = t0 - t1
        self.exponent = exponent

    def temperature(self, degree_of_completion):
        return self.t0 - self.tDrop * (math.pow(degree_of_completion, self.exponent))

    def _get_params(self):
        return {"t0": self.t0, "t1": self.t1, "exponent": self.exponent}


class SAProbabilityFunction(ABC):
    def __init__(self, **params_dict):
        self.params_dict = params_dict

    def __str__(self):
        return f"{self.__class__.__name__}{self.params_dict}"

    @abstractmethod
    def __call__(self, degree_of_completion):
        pass


class SAProbabilityFunctionLinear(SAProbabilityFunction):
    """A probability function where probabilities decay linearly"""
    def __init__(self, p0=1.0, p1=0.0):
        super().__init__(p0=p0, p1=p1)
        self.p0 = p0
        self.p1 = p1

    def __call__(self, degree_of_completion):
        return self.p0 - degree_of_completion * (self.p0 - self.p1)


class SAProbabilityFunctionReverseSigmoid(SAProbabilityFunction):
    """A probability function where probabilities decay in a reverse sigmoid shape"""
    def __init__(self, p0=1.0, p1=0.0, steepness=10):
        """
        :param p0: the probability at the beginning (degree of completion 0)
        :param p1: the probability at the end (degree of completion 1)
        :param steepness: the steepness of the sigmoid curve
        """
        super().__init__(p0=p0, p1=p1, steepness=steepness)
        self.p0 = p0
        self.p1 = p1
        self.steepness = steepness

    def __call__(self, degree_of_completion):
        return reverse_sigmoid(degree_of_completion, self.p0, self.p1, self.steepness)


class SAProbabilityFunctionConstant(SAProbabilityFunction):
    """A constant probability function (which returns the same probability for all degrees of completion)"""
    def __init__(self, p):
        super().__init__(p=p)
        self.p = p

    def __call__(self, degree_of_completion):
        return self.p


class SAProbabilitySchedule(SATemperatureSchedule):
    """
    A temperature schedule where temperatures are derived from a probability schedule that is to apply to a reference cost delta, which
    is either given or is computed from observed values (the latter resulting in an adaptive schedule).
    It converts a function that returns probabilities for degrees of completion into a corresponding temperature schedule.
    """
    def __init__(self, reference_cost_delta: Optional[float], probability_function: SAProbabilityFunction):
        """
        Creates a temperature schedule for a reference cost delta (which can also be computed automatically from observed data)
        and probability function:
        The schedule will return temperatures such that for referenceCostDelta, the probability of acceptance of a move at
        degree of completion d in [0,1] will be probabilityFunction(d).

        :param reference_cost_delta: the (positive) cost delta for which the probability function is to apply;
            if None, adaptively determine it from the empirical mean
        :param probability_function: a function which maps degrees of completion in [0,1] to probabilities in [0,1]
        """
        self.adaptive = reference_cost_delta is None
        self.referenceCostDelta = reference_cost_delta
        self.probabilityFunction = probability_function
        self.paramsDict = {
            "refCostDelta": reference_cost_delta if reference_cost_delta is not None else "adaptive",
            "probabilityFunction": str(probability_function)}

        # helper variables for adaptive mode
        self._costDeltaSum = 0
        self._costDeltaCount = 0

    def temperature(self, degree_of_completion):
        if self.adaptive and self._costDeltaCount == 0:
            raise Exception("Cannot generate a temperature from an adaptive schedule without any previous cost-delta samples")
        p = self.probabilityFunction(degree_of_completion)
        if p == 0.0:
            return 0
        else:
            return -self.referenceCostDelta / math.log(p)

    def probability(self, degree_of_completion, cost_delta):
        if self.adaptive:
            self._costDeltaSum += cost_delta
            self._costDeltaCount += 1
            self.referenceCostDelta = max(self._costDeltaSum / self._costDeltaCount, 1e-10)
        return super().probability(degree_of_completion, cost_delta)

    def _get_params(self):
        return self.paramsDict


class SACostValue(ABC):
    """Representation of an immutable cost value"""

    @abstractmethod
    def value(self):
        """Returns the numeric cost value"""
        pass

    @abstractmethod
    def add(self, other) -> 'SACostValue':
        pass


class SACostValueNumeric(SACostValue):
    def __init__(self, scalar):
        self._value = scalar

    def __str__(self):
        return str(self.value())

    def value(self):
        return self._value

    def add(self, other: 'SACostValueNumeric') -> 'SACostValueNumeric':
        return SACostValueNumeric(self._value + other._value)


class SAState(ABC):
    """Represents the state/variable assignment during a simulated annealing process"""

    def __init__(self, r: random.Random):
        self.r = r
        self.cost = self.compute_cost_value()

    @abstractmethod
    def compute_cost_value(self) -> SACostValue:
        """Computes the costs of this state (from scratch)"""
        pass

    @abstractmethod
    def get_state_representation(self):
        """
        Returns a compact state representation (for the purpose of archiving a hitherto best result), which can later be
        applied via applyStateRepresentation.

        :return: a compact state representation of some sort
        """
        pass

    @abstractmethod
    def apply_state_representation(self, representation):
        """
        Applies the given state representation (as returned via `getStateRepresentation`) in order for the optimisation result to
        be obtained by the user.
        Note that the function does not necessarily need to change this state to reflect the representation, as its sole
        purpose is for the optimsation result to be obtainable thereafter (it is not used during the optimisation process as such).

        :param representation: a representation as returned by `getStateRepresentation`
        """
        pass


TSAState = TypeVar("TSAState", bound=SAState)


class SAOperator(Generic[TSAState]):
    """
    An operator which, when applied with appropriately chosen parameters, can transform a state into another
    state during simulated annealing
    """

    def __init__(self, state: TSAState):
        """
        :param state: the state to which the operator is applied
        """
        self.state = state

    def apply_cost_change(self, cost_delta: SACostValue):
        """
        Applies the cost change to the state given at construction

        :param cost_delta: the cost change to apply
        """
        self.state.cost = self.state.cost.add(cost_delta)

    @abstractmethod
    def apply_state_change(self, *params):
        """
        Applies the operator to the state, i.e. it makes the changes to the state only
        (and does not update the associated costs)

        :param params: the parameters with which the operator is to be applied
        :return:
        """
        pass

    def apply(self, params: Tuple, cost_delta: SACostValue):
        """
        Applies the operator to the state given at construction, changing the state and updating the costs appropriately

        :param params: the parameters with which the operator is to be applied
        :param cost_delta: the cost change that results from the application
        :return:
        """
        self.apply_cost_change(cost_delta)
        self.apply_state_change(*params)

    @abstractmethod
    def cost_delta(self, *params) -> SACostValue:
        """
        Computes the cost change that would apply when applying this operator with the given parameters

        :param params: an arbitrary list of parameters (specific to the concrete operator)
        :return:
        """
        pass

    @abstractmethod
    def choose_params(self) -> Optional[Tuple[Tuple, Optional[SACostValue]]]:
        """
        Chooses parameters for the application of the operator (e.g. randomly or greedily).

        :return: a tuple (params, costValue) or None if no suitable parameters are found, where params is the list of chosen
            parameters and costValue is either an instance of CostValue or None if the costs have not been computed.
        """
        pass


class SAChain(Generic[TSAState]):
    """Manages the progression of one state during simulated annealing"""

    log = log.getChild(__qualname__)

    def __init__(self,
            state_factory: Callable[[random.Random], TSAState],
            schedule: SATemperatureSchedule,
            ops_and_weights: Sequence[Tuple[Callable[[TSAState], SAOperator[TSAState]], float]],
            random_seed,
            collect_stats=False):
        self.schedule = schedule
        self.r = random.Random(random_seed)
        self.state = state_factory(self.r)
        self.collect_stats = collect_stats
        operators, weights = zip(*ops_and_weights)
        cum_weights, s = [], 0
        for weight in weights:
            s += weight
            cum_weights.append(s)
        self.ops = [cons(self.state) for cons in operators]
        self.op_cum_weights = cum_weights
        self.steps_taken = 0
        self.count_none_params = 0
        self.count_best_updates = -1
        self.best_cost = None
        self.best_state_repr = None
        self.logged_series = collections.defaultdict(lambda: [])
        self._update_best_state()

        if self.collect_stats:
            self.operator_inapplicability_counters = {}
            for op in self.ops:
                self.operator_inapplicability_counters[op] = RelativeFrequencyCounter()

    def _update_best_state(self):
        cost = self.state.cost
        if self.best_cost is None or cost.value() < self.best_cost.value():
            self.best_cost = cost
            self.best_state_repr = self.state.get_state_representation()
            self.count_best_updates += 1

    def step(self, degree_of_completion):
        r = self.r

        # make move
        op = r.choices(self.ops, cum_weights=self.op_cum_weights, k=1)[0]
        param_choice = op.choose_params()
        if param_choice is None:
            self.count_none_params += 1
        else:
            params, cost_change = param_choice
            if cost_change is None:
                cost_change = op.cost_delta(*params)
            if cost_change.value() < 0:
                make_move = True
            else:
                cost_change_value = cost_change.value()
                p, T = self.schedule.probability(degree_of_completion, cost_change_value)
                make_move = r.random() <= p
                self.log.debug(f'p: {p}, T: {T}, costDelta: {cost_change_value}, move: {make_move}')
                if self.collect_stats:
                    self.logged_series["temperatures"].append(T)
                    self.logged_series["probabilities"].append(p)
            if make_move:
                op.apply(params, cost_change)
                self._update_best_state()
            if self.collect_stats:
                self.logged_series["costDeltas"].append(cost_change.value())
        if self.collect_stats:
            self.logged_series["bestCostValues"].append(self.best_cost.value())
            self.logged_series["costValues"].append(self.state.cost.value())
            self.operator_inapplicability_counters[op].count(param_choice is None)

        self.steps_taken += 1

        if self.log.isEnabledFor(logging.DEBUG):
            self.log.debug(f"Step {self.steps_taken}: cost={self.state.cost}; best cost={self.best_cost}")

    def log_stats(self):
        if self.collect_stats:
            stats = {"useless moves total (None params)": f"{self.count_none_params}/{self.steps_taken}"}
            for op, counter in self.operator_inapplicability_counters.items():
                stats[f"useless moves of {op}"] = str(counter)
            logged_cost_deltas = self.logged_series["costDeltas"]
            if logged_cost_deltas:
                stats["mean cost delta"] = f"{np.mean(logged_cost_deltas):.3f} +- { np.std(logged_cost_deltas):.3f}"
                abs_cost_deltas = np.abs(logged_cost_deltas)
                stats["mean absolute cost delta"] = f"{np.mean(abs_cost_deltas):.3f} +- {np.std(abs_cost_deltas):.3f}"
                positive_cost_deltas = [cd for cd in logged_cost_deltas if cd > 0]
                if positive_cost_deltas:
                    stats["positive cost delta"] = f"mean={np.mean(positive_cost_deltas):.3f} +- {np.std(positive_cost_deltas):.3f}," \
                                                   f" max={np.max(positive_cost_deltas):.3f}"
            stats_join = "\n    "
            self.log.info(f"Stats: {stats_join.join([key + ': ' + value for (key, value) in stats.items()])}")
        self.log.info(f"Best solution has {self.best_cost} after {self.count_best_updates} updates of best state")

    def apply_best_state(self):
        """Applies the best state representation found in this chain to the chain's state"""
        self.state.apply_state_representation(self.best_state_repr)
        self.state.cost = self.best_cost

    def plot_series(self, series_name):
        """
        Plots one of the logged series

        :param series_name: the name of the series (see getSeries)
        """
        series = self.get_series(series_name)
        plt.figure()
        series.plot(title=series_name)

    def get_series(self, series_name):
        """
        Gets one of the logged series (for collectStats==True)

        :param series_name: name of the series: one of "temperatures", "probabilities", "costDeltas", "bestCostValues", "costValues
        """
        if not self.collect_stats:
            raise Exception("No stats were collected")
        if series_name not in self.logged_series:
            raise Exception("Unknown series")
        return pd.Series(self.logged_series[series_name])


class SimulatedAnnealing(Generic[TSAState]):
    """
    The simulated annealing algorithm for discrete optimisation (cost minimisation)
    """

    log = log.getChild(__qualname__)

    def __init__(self,
            schedule_factory: Callable[[], SATemperatureSchedule],
            ops_and_weights: Sequence[Tuple[Callable[[TSAState], SAOperator[TSAState]], float]],
            max_steps: int = None,
            duration: float = None,
            random_seed=42,
            collect_stats=False):
        """
        :param schedule_factory: a factory for the creation of the temperature schedule for the annealing process
        :param ops_and_weights: a list of operator factories with associated weights, where weights are to indicate the (non-normalised)
            probability of choosing the associated operator
        :param max_steps: the number of steps for which to run the optimisation; may be None (if not given, duration must be provided)
        :param duration: the duration, in seconds, for which to run the optimisation; may be None (if not given, maxSteps must be provided)
        :param random_seed: the random seed to use for all random choices
        :param collect_stats: flag indicating whether to collect additional statics which will be logged
        """
        if max_steps is not None and max_steps <= 0:
            raise ValueError("The number of iterations should be greater than 0.")
        if max_steps is None and duration is None or (max_steps is not None and duration is not None):
            raise ValueError("Exactly one of {maxSteps, duration} must be specified.")
        if duration is not None and duration <= 0:
            raise ValueError("Duration must be greater than 0 if provided")
        self.scheduleFactory = schedule_factory
        self.max_steps = max_steps
        self.duration = duration
        self.randomSeed = random_seed
        self.opsAndWeights = ops_and_weights
        self.collect_stats = collect_stats
        self._chain = None

    def optimise(self, state_factory: Callable[[random.Random], TSAState]) -> TSAState:
        """
        Applies the annealing process, starting with a state created via the given factory.

        :param state_factory: the factory with which to create the (initial) state
        :return: the state with the least-cost representation found during the optimisation applied
        """
        chain = SAChain(state_factory, self.scheduleFactory(), ops_and_weights=self.opsAndWeights, random_seed=self.randomSeed,
            collect_stats=self.collect_stats)
        self.log.info(f"Running simulated annealing with {len(self.opsAndWeights)} operators for "
                      f"{'%d steps' % self.max_steps if self.max_steps is not None else '%d seconds' % self.duration} ...")
        start_time = time.time()
        while True:
            time_elapsed = time.time() - start_time
            if (self.max_steps is not None and chain.steps_taken >= self.max_steps) or (self.duration is not None and time_elapsed >= self.duration):
                break
            if self.max_steps is not None:
                degree_of_completion = chain.steps_taken / self.max_steps
            else:
                degree_of_completion = time_elapsed / self.duration
            chain.step(degree_of_completion)
        self.log.info(f"Simulated annealing completed after {time.time()-start_time:.1f} seconds, {chain.steps_taken} steps")
        chain.log_stats()
        chain.apply_best_state()
        if self.collect_stats:
            self._chain = chain
        return chain.state

    def get_chain(self) -> Optional[SAChain[TSAState]]:
        """
        Gets the chain used by the most recently completed application (optimise call) of this object
        for the case where stats collection was enabled; the chain then contains relevant series and may be used
        to generate plots. If stats collection was not enabled, returns None.
        """
        return self._chain


class ParallelTempering(Generic[TSAState]):
    """
    The parallel tempering algorithm for discrete optimisation (cost minimisation)
    """

    log = log.getChild(__qualname__)

    def __init__(self,
            num_chains,
            ops_and_weights: Sequence[Tuple[Callable[[TSAState], SAOperator[TSAState]], float]],
            schedule: SATemperatureSchedule = None,
            probability_function: SAProbabilityFunction = None,
            max_steps: int = None,
            duration: float = None,
            random_seed=42,
            log_cost_progression=False):
        """
        Creates a parallel tempering optimiser with the given number of chains and operators for each chain.
        To determine the schedule to use for each chain, either schedule or probabilityFunction must be provided.
        It is usually more robust to use adaptive schedules and therefore to provide probabilityFunction.

        :param num_chains: the number of parallel chains
        :param ops_and_weights: a list of operators with associated weights (which are to indicate the non-normalised probability of
            choosing the associated operator)
        :param schedule: the temperature schedule from which numChains temperatures of chains are drawn (using equidistant degrees of
            completion); if None, must provide probabilityFunction
        :param probability_function: the probability function from which numChains probabilities for adaptive probability schedules, each
            using a constant probability, are to be drawn; if None, must provide schedule
        :param max_steps: the number of steps for which to run the optimisation; may be None (if not given, duration must be provided)
        :param duration: the duration, in seconds, for which to run the optimisation; may be None (if not given, maxSteps must be provided)
        :param random_seed: the random seed to use for all random choices
        :param log_cost_progression: whether to log cost progression of all chains (such that it can be plotted after the fact via
            plotCostProgression)
        """
        if max_steps is not None and max_steps <= 0:
            raise ValueError("The number of iterations should be greater than 0.")
        if (max_steps is None and duration is None) or (max_steps is not None and duration is not None):
            raise ValueError("Exactly one of {maxSteps, duration} must be specified.")
        if duration is not None and duration <= 0:
            raise ValueError("duration should be greater than 0 if provided.")
        if num_chains < 2:
            raise ValueError("Number of chains must be at least 2.")
        if (schedule is None and probability_function is None) or (schedule is not None and probability_function is not None):
            raise ValueError("Exactly one of {schedule, probabilityFunction} must be given")
        self.max_steps = max_steps
        self.duration = duration
        self.random_seed = random_seed
        self.num_chains = num_chains
        self.base_schedule = schedule
        self.base_probability_function = probability_function
        self.ops_and_weights = ops_and_weights
        self.log_cost_progression = log_cost_progression

        # transient members
        self._cost_progressions = None
        self._schedule_param_strings = None

    def _create_schedules(self):
        degree_step = 1.0 / (self.num_chains - 1)
        degrees_of_completion = [i*degree_step for i in range(self.num_chains)]
        if self.base_schedule is not None:
            # create schedules with fixed temperatures taken from base schedule
            temperatures = [self.base_schedule.temperature(d) for d in degrees_of_completion]
            self._schedule_param_strings = ["T=%.2f" % t for t in temperatures]
            return [SATemperatureScheduleFixed(t) for t in temperatures]
        else:
            # create adaptive probability schedules based on probabilities taken from base probability function
            probabilities = [self.base_probability_function(d) for d in degrees_of_completion]
            self._schedule_param_strings = ["p=%.3f" % p for p in probabilities]
            return [SAProbabilitySchedule(None, SAProbabilityFunctionConstant(p)) for p in probabilities]

    def optimise(self, state_factory: Callable[[random.Random], SAState]) -> SAState:
        """
        Applies the optimisation process, starting, in each chain, with a state created via the given factory.

        :param state_factory: the factory with which to create the states for all chains
        :return: the state with the least-cost representation found during the optimisation (among all parallel chains) applied
        """
        self.log.info(f"Running parallel tempering with {self.num_chains} chains, {len(self.ops_and_weights)} operators for "
                      f"{'%d steps' % self.max_steps if self.max_steps is not None else '%d seconds' % self.duration} ...")

        r = random.Random(self.random_seed)
        chains = []
        cost_progressions = []
        for i, schedule in enumerate(self._create_schedules(), start=1):
            self.log.info(f"Chain {i} uses {schedule}")
            chains.append(SAChain(state_factory, schedule, ops_and_weights=self.ops_and_weights, random_seed=r.randint(0, 1000)))
            cost_progressions.append([])

        start_time = time.time()
        step = 0
        num_chain_swaps = 0
        while True:
            time_elapsed = time.time() - start_time
            if (self.max_steps is not None and step > self.max_steps) or (self.duration is not None and time_elapsed > self.duration):
                break

            # take one step in each chain
            degree_of_completion = step / self.max_steps if self.max_steps is not None else time_elapsed / self.duration
            for chain in chains:
                chain.step(degree_of_completion)

            # check if neighbouring chains can be "swapped": if a high-temperature chain has a better state
            # than a low-temperature chain, swap them (by exchanging their schedules and swapping them
            # in the chains array, which shall always be in descending order of temperature)
            for idx_high_temp_chain in range(0, self.num_chains - 1):
                idx_low_temp_chain = idx_high_temp_chain+1
                high_temp_chain = chains[idx_high_temp_chain]
                low_temp_chain = chains[idx_low_temp_chain]
                if high_temp_chain.state.cost.value() < low_temp_chain.state.cost.value():
                    high_temp_chain.schedule, low_temp_chain.schedule = low_temp_chain.schedule, high_temp_chain.schedule
                    chains[idx_low_temp_chain] = high_temp_chain
                    chains[idx_high_temp_chain] = low_temp_chain
                    num_chain_swaps += 1

            if self.log_cost_progression:
                for idx_chain, chain in enumerate(chains):
                    cost_progressions[idx_chain].append(chain.state.cost.value())

            step += 1

        self.log.info(f"Number of chain swaps: {num_chain_swaps}")
        if self.log_cost_progression: self._cost_progressions = cost_progressions

        # apply best solution
        best_chain_idx = int(np.argmin([chain.best_cost.value() for chain in chains]))
        chains[best_chain_idx].apply_best_state()
        return chains[best_chain_idx].state

    def plot_cost_progression(self):
        if not self.log_cost_progression or self._cost_progressions is None:
            raise Exception("No cost progression was logged")
        series = {}
        for scheduleParamStr, costProgression in zip(self._schedule_param_strings, self._cost_progressions):
            series[scheduleParamStr] = costProgression
        plt.figure()
        pd.DataFrame(series).plot()
