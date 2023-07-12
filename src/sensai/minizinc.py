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

    def __init__(self, costValues: List[float], significantDigits: int):
        """
        Parameters:
            costValues: the sequence of cost values whose precision should be maintained in the int realm
            significantDigits: the number of significant digits that shall at least be maintained
        """
        exp10 = significantDigits - 1 - min([0] + [np.floor(np.log10(v)) for v in costValues])
        self.scalingFactor = math.pow(10, exp10)

    def scaledInt(self, originalValue: float) -> int:
        """Returns the scaled value as an integer"""
        return int(round(originalValue * self.scalingFactor))

    def scaledFloat(self, originalValue: float) -> float:
        return originalValue * self.scalingFactor

    def originalValue(self, scaledValue: float) -> float:
        """Returns the original unscaled value from a scaled value"""
        return scaledValue / self.scalingFactor

    def __str__(self):
        return "CostScaler[factor=%d]" % self.scalingFactor


class MiniZincProblem(ABC):

    def createMiniZincFile(self, f):
        """
        Writes MiniZinc code

        :param f: an OS-level handle to an open file
        """
        os.write(f, bytes(self.getMiniZincCode(), 'utf-8'))

    @abstractmethod
    def getMiniZincCode(self):
        pass


class MiniZincSolver(object):
    log = log.getChild(__qualname__)

    def __init__(self, name='OSICBC', solverTimeSeconds=None, fznOutputPath=None):
        """
        :param name: name of solver compatible with miniZinc
        :param solverTimeSeconds: upper time limit for solver in seconds
        :param fznOutputPath: flatZinc output path
        """
        self.solverName = name
        self.solverTimeLimitSecs = solverTimeSeconds
        self.fznOutputPath = fznOutputPath
        self.lastSolverTimeSecs = None
        self.lastSolverOutput = None
        self.lastSolverErrOutput = None

    def __str(self):
        return f"MiniZincSolver[{self.solverName}]"

    def solvePath(self, mznPath: str, logInfo=True) -> str:
        """
        Solves the MiniZinc problem stored at the given file path

        :param mznPath: path to file containing MiniZinc problem code
        :param logInfo: whether to log solver output at INFO level rather than DEBUG level
        :return: the solver output
        """
        self.lastSolverTimeSecs = None
        logSolver = self.log.info if logInfo else self.log.debug

        args = ["--statistics", "--solver", self.solverName]
        if self.solverTimeLimitSecs is not None:
            args.append("--time-limit")
            args.append(str(self.solverTimeLimitSecs * 1000))
        if self.fznOutputPath is not None:
            args.append("--output-fzn-to-file")
            args.append(self.fznOutputPath)
        args.append(mznPath)
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
            logSolver("Solver output: %s" % line.rstrip())
        output = "".join(output)
        proc.wait()
        if proc.returncode != 0:
            raise Exception(f"MiniZinc call failed with return code {proc.returncode}; output: {output}")
        self.lastSolverTimeSecs = time.time() - start_time
        self.lastSolverOutput = output
        self.log.info("Solver time: %.1fs" % self.lastSolverTimeSecs)
        return output

    def solveProblem(self, problem: MiniZincProblem, keepTempFile=False, logInfo=True) -> str:
        """
        Solves the given MiniZinc problem

        :param problem: the problem to solve
        :param keepTempFile: whether to keep the temporary .mzv file
        :param logInfo: whether to log solver output at INFO level rather than DEBUG level
        :return: the solver output
        """
        f, path = tempfile.mkstemp(".mzn")
        try:
            try:
                problem.createMiniZincFile(f)
            finally:
                os.close(f)
            return self.solvePath(path, logInfo=logInfo)
        finally:
            if not keepTempFile:
                os.unlink(path)

    def getLastSolverTimeSecs(self):
        return self.lastSolverTimeSecs


def extract1DArrayFromOutput(stringIdentifier: str, output: str) -> List:
    regexOutput = re.search(r'{stringIdentifier} = array1d\(\d+\.\.\d+, (\[.*?\])'.format(stringIdentifier=stringIdentifier), output)
    return eval(regexOutput.group(1))


def extractMultiDimArrayFromOutput(stringIdentifier: str, dim: int, output: str, boolean=False) -> np.array:
    dimRegex = r"1..(\d+), "
    regex = r'{stringIdentifier} = array{dim}d\({dimsRegex}(\[.*?\])'.format(stringIdentifier=stringIdentifier, dim=dim, dimsRegex=dimRegex*dim)
    match = re.search(regex, output)
    if match is None:
        raise Exception("No match found for regex: %s" % regex)
    shape = [int(match.group(i)) for i in range(1, dim+1)]
    flatList = match.group(dim+1)
    if boolean:
        flatList = flatList.replace("false", "0").replace("true", "1")
    flatList = eval(flatList)
    array1d = np.array(flatList)
    arraymd = array1d.reshape(shape)
    return arraymd


def array2MiniZinc(a: np.array, elementCast):
    shape = a.shape
    dims = ", ".join([f"1..{n}" for n in shape])
    values = str(list(map(elementCast, a.flatten())))
    return f"array{len(shape)}d({dims}, {values})"
