import io
import logging
import os
from typing import Sequence, Optional, Tuple, List

import matplotlib.figure
from matplotlib import pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)


class ResultWriter:
    log = log.getChild(__qualname__)

    def __init__(self, resultDir, filenamePrefix=""):
        self.resultDir = resultDir
        os.makedirs(resultDir, exist_ok=True)
        self.filenamePrefix = filenamePrefix

    def childWithAddedPrefix(self, prefix: str) -> "ResultWriter":
        """
        Creates a derived result writer with an added prefix, i.e. the given prefix is appended to this
        result writer's prefix

        :param prefix: the prefix to append
        :return: a new writer instance
        """
        return ResultWriter(self.resultDir, filenamePrefix=self.filenamePrefix + prefix)

    def childForSubdirectory(self, dirName: str):
        resultDir = os.path.join(self.resultDir, dirName)
        os.makedirs(resultDir, exist_ok=True)
        return ResultWriter(resultDir, filenamePrefix=self.filenamePrefix)

    def path(self, filenameSuffix: str, extensionToAdd=None, validOtherExtensions: Optional[Sequence[str]] = None):
        """
        :param filenameSuffix: the suffix to add (which may or may not already include a file extension)
        :param extensionToAdd: if not None, the file extension to add (without the leading ".") unless
            the extension to add or one of the extenions in validExtensions is already present
        :param validOtherExtensions: a sequence of valid other extensions (without the "."), only
            relevant if extensionToAdd is specified
        :return: the full path
        """
        if extensionToAdd is not None:
            addExt = True
            validExtensions = set(validOtherExtensions) if validOtherExtensions is not None else set()
            validExtensions.add(extensionToAdd)
            if validExtensions is not None:
                for ext in validExtensions:
                    if filenameSuffix.endswith("." + ext):
                        addExt = False
                        break
            if addExt:
                filenameSuffix += "." + extensionToAdd
        path = os.path.join(self.resultDir, f"{self.filenamePrefix}{filenameSuffix}")
        return path

    def writeTextFile(self, filenameSuffix, content):
        p = self.path(filenameSuffix, extensionToAdd="txt")
        self.log.info(f"Saving text file {p}")
        with open(p, "w") as f:
            f.write(content)
        return p

    def writeTextFileLines(self, filenameSuffix, lines: List[str]):
        p = self.path(filenameSuffix, extensionToAdd="txt")
        self.log.info(f"Saving text file {p}")
        writeTextFileLines(lines, p)
        return p

    def writeDataFrameTextFile(self, filenameSuffix, df: pd.DataFrame):
        p = self.path(filenameSuffix, extensionToAdd="df.txt", validOtherExtensions="txt")
        self.log.info(f"Saving data frame text file {p}")
        with open(p, "w") as f:
            f.write(df.to_string())
        return p

    def writeDataFrameCsvFile(self, filenameSuffix, df: pd.DataFrame):
        p = self.path(filenameSuffix, extensionToAdd="csv")
        self.log.info(f"Saving data frame CSV file {p}")
        df.to_csv(p)
        return p

    def writeFigure(self, filenameSuffix, fig, closeFigure=False):
        """
        :param filenameSuffix: the filename suffix, which may or may not include a file extension, valid extensions being {"png", "jpg"}
        :param fig: the figure to save
        :param closeFigure: whether to close the figure after having saved it
        :return: the path to the file that was written
        """
        p = self.path(filenameSuffix, extensionToAdd="png", validOtherExtensions=("jpg",))
        self.log.info(f"Saving figure {p}")
        fig.savefig(p)
        if closeFigure:
            plt.close(fig)
        return p

    def writeFigures(self, figures: Sequence[Tuple[str, matplotlib.figure.Figure]], closeFigures=False):
        for name, fig in figures:
            self.writeFigure(name, fig, closeFigure=closeFigures)

    def writePickle(self, filenameSuffix, obj):
        from .pickle import dumpPickle
        p = self.path(filenameSuffix, extensionToAdd="pickle")
        self.log.info(f"Saving pickle {p}")
        dumpPickle(obj, p)
        return p


def writeTextFileLines(lines: List[str], path):
    """
    :param lines: the lines to write (without a trailing newline, which will be added)
    :param path: the path of the text file to write to
    """
    with open(path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def readTextFileLines(path, strip=True, skipEmpty=True) -> List[str]:
    """
    :param path: the path of the text file to read from
    :param strip: whether to strip each line, removing whitespace/newline characters
    :param skipEmpty: whether to skip any lines that are empty (after stripping)
    :return: the list of lines
    """
    lines = []
    with open(path, "r") as f:
        for line in f.readlines():
            if strip:
                line = line.strip()
            if not skipEmpty or line != "":
                lines.append(line)
    return lines


def isS3Path(path: str):
    return path.startswith("s3://")


class S3Object:
    def __init__(self, path):
        assert isS3Path(path)
        self.path = path
        self.bucket, self.object = self.path[5:].split("/", 1)

    class OutputFile:
        def __init__(self, s3Object: "S3Object"):
            self.s3Object = s3Object
            self.buffer = io.BytesIO()

        def write(self, obj: bytes):
            self.buffer.write(obj)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.s3Object.put(self.buffer.getvalue())

    def getFileContent(self):
        return self._getS3Object().get()['Body'].read()

    def openFile(self, mode):
        assert mode in ("wb", "rb")
        if mode == "rb":
            content = self.getFileContent()
            return io.BytesIO(content)

        elif mode == "wb":
            return self.OutputFile(self)

        else:
            raise ValueError(mode)

    def put(self, obj: bytes):
        self._getS3Object().put(Body=obj)

    def _getS3Object(self):
        import boto3
        session = boto3.session.Session(profile_name=os.getenv("AWS_PROFILE"))
        s3 = session.resource("s3")
        return s3.Bucket(self.bucket).Object(self.object)

