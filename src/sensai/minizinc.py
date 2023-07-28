from abc import ABC, abstractmethod
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from typing import List

import numpy as np


log = logging.getLogger(__name__)


class CostScaler:
    """
    Serves to scale floats and converts them into integers (and vice versa) whilst
    maintaining decimal precision
    """

    def __init__(self, cost_values: List[float], significant_digits: int):
        """
        Parameters:
            cost_values: the sequence of cost values whose precision should be maintained in the int realm
            significant_digits: the number of significant digits that shall at least be maintained
        """
        exp10 = significant_digits - 1 - min([0] + [np.floor(np.log10(v)) for v in cost_values])
        self.scalingFactor = math.pow(10, exp10)

    def scaled_int(self, original_value: float) -> int:
        """Returns the scaled value as an integer"""
        return int(round(original_value * self.scalingFactor))

    def scaled_float(self, original_value: float) -> float:
        return original_value * self.scalingFactor

    def original_value(self, scaled_value: float) -> float:
        """Returns the original unscaled value from a scaled value"""
        return scaled_value / self.scalingFactor

    def __str__(self):
        return "CostScaler[factor=%d]" % self.scalingFactor


class MiniZincProblem(ABC):

    def create_mini_zinc_file(self, f):
        """
        Writes MiniZinc code

        :param f: an OS-level handle to an open file
        """
        os.write(f, bytes(self.get_mini_zinc_code(), 'utf-8'))

    @abstractmethod
    def get_mini_zinc_code(self):
        pass


class MiniZincSolver(object):
    log = log.getChild(__qualname__)

    def __init__(self, name='OSICBC', solver_time_seconds=None, fzn_output_path=None):
        """
        :param name: name of solver compatible with miniZinc
        :param solver_time_seconds: upper time limit for solver in seconds
        :param fzn_output_path: flatZinc output path
        """
        self.solver_name = name
        self.solver_time_limit_secs = solver_time_seconds
        self.fzn_output_path = fzn_output_path
        self.last_solver_time_secs = None
        self.last_solver_output = None
        self.lastSolverErrOutput = None

    def __str(self):
        return f"MiniZincSolver[{self.solver_name}]"

    def solve_path(self, mzn_path: str, log_info=True) -> str:
        """
        Solves the MiniZinc problem stored at the given file path

        :param mzn_path: path to file containing MiniZinc problem code
        :param log_info: whether to log solver output at INFO level rather than DEBUG level
        :return: the solver output
        """
        self.last_solver_time_secs = None
        log_solver = self.log.info if log_info else self.log.debug

        args = ["--statistics", "--solver", self.solver_name]
        if self.solver_time_limit_secs is not None:
            args.append("--time-limit")
            args.append(str(self.solver_time_limit_secs * 1000))
        if self.fzn_output_path is not None:
            args.append("--output-fzn-to-file")
            args.append(self.fzn_output_path)
        args.append(mzn_path)
        command = "minizinc " + " ".join(args)

        self.log.info("Running %s" % command)
        start_time = time.time()
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = []
        while True:
            line = proc.stdout.readline().decode("utf-8")
            if not line:
                break
            output.append(line)
            log_solver("Solver output: %s" % line.rstrip())
        output = "".join(output)
        proc.wait()
        if proc.returncode != 0:
            raise Exception(f"MiniZinc call failed with return code {proc.returncode}; output: {output}")
        self.last_solver_time_secs = time.time() - start_time
        self.last_solver_output = output
        self.log.info("Solver time: %.1fs" % self.last_solver_time_secs)
        return output

    def solve_problem(self, problem: MiniZincProblem, keep_temp_file=False, log_info=True) -> str:
        """
        Solves the given MiniZinc problem

        :param problem: the problem to solve
        :param keep_temp_file: whether to keep the temporary .mzv file
        :param log_info: whether to log solver output at INFO level rather than DEBUG level
        :return: the solver output
        """
        f, path = tempfile.mkstemp(".mzn")
        try:
            try:
                problem.create_mini_zinc_file(f)
            finally:
                os.close(f)
            return self.solve_path(path, log_info=log_info)
        finally:
            if not keep_temp_file:
                os.unlink(path)

    def get_last_solver_time_secs(self):
        return self.last_solver_time_secs


def extract_1d_array_from_output(string_identifier: str, output: str) -> List:
    regexOutput = re.search(r'{stringIdentifier} = array1d\(\d+\.\.\d+, (\[.*?\])'.format(stringIdentifier=string_identifier), output)
    return eval(regexOutput.group(1))


def extract_multi_dim_array_from_output(string_identifier: str, dim: int, output: str, boolean=False) -> np.array:
    dim_regex = r"1..(\d+), "
    regex = r'{stringIdentifier} = array{dim}d\({dimsRegex}(\[.*?\])'.format(stringIdentifier=string_identifier, dim=dim,
        dimsRegex=dim_regex*dim)
    match = re.search(regex, output)
    if match is None:
        raise Exception("No match found for regex: %s" % regex)
    shape = [int(match.group(i)) for i in range(1, dim+1)]
    flat_list = match.group(dim+1)
    if boolean:
        flat_list = flat_list.replace("false", "0").replace("true", "1")
    flat_list = eval(flat_list)
    array1d = np.array(flat_list)
    arraymd = array1d.reshape(shape)
    return arraymd


def array_to_mini_zinc(a: np.array, element_cast):
    shape = a.shape
    dims = ", ".join([f"1..{n}" for n in shape])
    values = str(list(map(element_cast, a.flatten())))
    return f"array{len(shape)}d({dims}, {values})"
