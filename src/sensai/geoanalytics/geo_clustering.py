import collections
import itertools
import math
from abc import abstractmethod, ABC
from typing import List, Tuple, Iterator, Optional

import numpy as np
import sklearn.cluster

from .geo_coords import GeoCoord
from .local_coords import LocalCoordinateSystem
from ..clustering import GreedyAgglomerativeClustering


class GeoCoordClusterer(ABC):
    @abstractmethod
    def fitGeoCoords(self, geoCoords: List[GeoCoord]):
        """
        :param geoCoords: the coordinates to be clustered
        """
        pass

    @abstractmethod
    def clustersIndices(self) -> Tuple[List[List[int]], List[int]]:
        """
        :return: a tuple (clusters, outliers), where clusters is a list of point indices, one list for each
            cluster containing the indices of points within the cluster, and outliers is the list of indices of points not within
            clusters
        """
        pass


class GreedyAgglomerativeGeoCoordClusterer(GeoCoordClusterer):
    def __init__(self, maxMinDistanceForMergeM: float, maxDistanceM: float, minClusterSize: int, lcs: LocalCoordinateSystem = None):
        """
        :param maxMinDistanceForMergeM: the maximum distance, in metres, for the minimum distance between two existing clusters for a merge
            to be admissible
        :param maxDistanceM: the maximum distance, in metres, between any two points for the points to be allowed to be in the same cluster
        :param minClusterSize: the minimum number of points any valid cluster must ultimately contain; the points in any smaller clusters
            shall be considered as outliers
        :param lcs: the local coordinate system to use for clustering; if None, compute based on mean coordinates passed when fitting
        """
        self.lcs = lcs
        self.minClusterSize = minClusterSize
        self.maxMinDistanceForMerge = maxMinDistanceForMergeM
        self.maxDistanceM = maxDistanceM
        self.squaredMaxMinDistanceForMerge = maxMinDistanceForMergeM * maxMinDistanceForMergeM
        self.squaredMaxDistance = maxDistanceM * maxDistanceM
        self.localPoints = None
        self.mMinDistance: Optional["GreedyAgglomerativeGeoCoordClusterer.Matrix"] = None
        self.mMaxSquaredDistance: Optional["GreedyAgglomerativeGeoCoordClusterer.Matrix"] = None

    class Matrix:
        UNSET_VALUE = np.inf

        def __init__(self, dim: int):
            self.m = np.empty((dim, dim))
            self.m.fill(np.inf)

        def set(self, c1: int, c2: int, value: float):
            self.m[c1][c2] = value
            self.m[c2][c1] = value

        def get(self, c1: int, c2: int) -> float:
            return self.m[c1][c2]

    class LocalPoint:
        def __init__(self, xy: np.ndarray, idx: int):
            self.idx = idx
            self.xy = xy

    class Cluster(GreedyAgglomerativeClustering.Cluster):
        def __init__(self, point: "GreedyAgglomerativeGeoCoordClusterer.LocalPoint", idx: int, clusterer: 'GreedyAgglomerativeGeoCoordClusterer'):
            self.idx = idx
            self.clusterer = clusterer
            self.points = [point]

        def mergeCost(self, other: "GreedyAgglomerativeGeoCoordClusterer.Cluster"):
            cartesianProduct = itertools.product(self.points, other.points)
            minSquaredDistance = math.inf
            maxSquaredDistance = 0
            for p1, p2 in cartesianProduct:
                diff = p1.xy - p2.xy
                squaredDistance = np.dot(diff, diff)
                if squaredDistance > self.clusterer.squaredMaxDistance:
                    maxSquaredDistance = math.inf
                    break
                else:
                    minSquaredDistance = min(squaredDistance, minSquaredDistance)

            # fill cache: the max value takes precedence; if it is inf (no merge admissible), then the min value is also set to inf;
            # the min value valid only if the max value is finite
            self.clusterer.mMaxSquaredDistance.set(self.idx, other.idx, maxSquaredDistance)
            if np.isinf(maxSquaredDistance):
                self.clusterer.mMinDistance.set(self.idx, other.idx, np.inf)
            else:
                self.clusterer.mMinDistance.set(self.idx, other.idx, minSquaredDistance)

            if np.isinf(maxSquaredDistance):
                return math.inf
            if minSquaredDistance <= self.clusterer.squaredMaxMinDistanceForMerge:
                return minSquaredDistance
            return math.inf

        def merge(self, other):
            self.points += other.points

    def fitGeoCoords(self, geoCoords: List[GeoCoord]) -> None:
        self.mMinDistance = self.Matrix(len(geoCoords))
        self.mMaxSquaredDistance = self.Matrix(len(geoCoords))
        if self.lcs is None:
            meanCoord = GeoCoord.meanCoord(geoCoords)
            self.lcs = LocalCoordinateSystem(meanCoord.lat, meanCoord.lon)
        self.localPoints = [self.LocalPoint(np.array(self.lcs.getLocalCoords(p.lat, p.lon)), idx) for idx, p in enumerate(geoCoords)]
        clusters = [self.Cluster(lp, i, self) for i, lp in enumerate(self.localPoints)]
        gac = GreedyAgglomerativeClustering(clusters,
            mergeCandidateDeterminationStrategy=self.MergeCandidateDeterminationStrategy(self.maxDistanceM, self))
        clusters = gac.applyClustering()
        self.clusters = clusters

    def clustersIndices(self) -> Tuple[List[List[int]], List[int]]:
        outliers = []
        clusters = []
        for c in self.clusters:
            indices = [p.idx for p in c.points]
            if len(c.points) < self.minClusterSize:
                outliers.extend(indices)
            else:
                clusters.append(indices)
        return clusters, outliers

    class MergeCandidateDeterminationStrategy(GreedyAgglomerativeClustering.MergeCandidateDeterminationStrategy):
        def __init__(self, searchRadiusM: float, parent: "GreedyAgglomerativeGeoCoordClusterer"):
            super().__init__()
            self.parent = parent
            self.searchRadiusM = searchRadiusM

        def setClusterer(self, clusterer: GreedyAgglomerativeClustering):
            super().setClusterer(clusterer)
            points = []
            for wc in self.clusterer.wrappedClusters:
                c: GreedyAgglomerativeGeoCoordClusterer.Cluster = wc.cluster
                for p in c.points:
                    points.append(p.xy)
            assert len(points) == len(self.clusterer.wrappedClusters)
            points = np.stack(points)
            self.kdtree = sklearn.neighbors.KDTree(points)

        def iterCandidateIndices(self, wc: "GreedyAgglomerativeClustering.WrappedCluster", initial: bool,
                mergedClusterIndices: Tuple[int, int] = None) -> Iterator[int]:
            c: GreedyAgglomerativeGeoCoordClusterer.Cluster = wc.cluster
            if initial:
                localPoint = c.points[0]  # pick any point from wc, since we use maximum cluster extension as search radius
                indices = self.kdtree.query_radius(np.reshape(localPoint.xy, (1, 2)), self.searchRadiusM)[0]
                candidateSet = set()
                for idx in indices:
                    wc = self.clusterer.wrappedClusters[idx]
                    candidateSet.add(wc.getClusterAssociation().idx)
                yield from sorted(candidateSet)
            else:
                # The new distance values (max/min) between wc and any cluster index otherIdx can be computed from the cached distance values
                # of the two clusters from which wc was created through a merge:
                # The max distance is the maximum of the squared distances of the original clusters (and if either is inf, then
                # a merge is definitely inadmissible, because one of the original clusters was already too far away).
                # The min distance is the minimum of the squred distances of the original clusters.
                c1, c2 = mergedClusterIndices
                max1 = self.parent.mMaxSquaredDistance.m[c1]
                max2 = self.parent.mMaxSquaredDistance.m[c2]
                maxCombined = np.maximum(max1, max2)
                for otherIdx, maxSqDistance in enumerate(maxCombined):
                    minSqDistance = np.inf
                    if maxSqDistance <= self.parent.squaredMaxDistance:
                        wcOther = self.clusterer.wrappedClusters[otherIdx]
                        if wcOther.isMerged():
                            continue
                        min1 = self.parent.mMinDistance.get(c1, otherIdx)
                        min2 = self.parent.mMinDistance.get(c2, otherIdx)
                        minSqDistance = min(min1, min2)
                        if minSqDistance <= self.parent.squaredMaxMinDistanceForMerge:
                            yield GreedyAgglomerativeClustering.ClusterMerge(wc, wcOther, minSqDistance)
                    # update cache
                    self.parent.mMaxSquaredDistance.set(wc.idx, otherIdx, maxSqDistance)
                    self.parent.mMinDistance.set(wc.idx, otherIdx, minSqDistance)


class SkLearnGeoCoordClusterer(GeoCoordClusterer):
    def __init__(self, clusterer, lcs: LocalCoordinateSystem = None):
        """
        :param clusterer: a clusterer from sklearn.cluster
        :param lcs: the local coordinate system to use for Euclidian conversion; if None, determine from data (using mean coordinate as centre)
        """
        self.lcs = lcs
        self.clusterer = clusterer
        self.localPoints = None

    def fitGeoCoords(self, geoCoords: List[GeoCoord]):
        if self.lcs is None:
            meanCoord = GeoCoord.meanCoord(geoCoords)
            self.lcs = LocalCoordinateSystem(meanCoord.lat, meanCoord.lon)
        self.localPoints = [self.lcs.getLocalCoords(p.lat, p.lon) for p in geoCoords]
        self.clusterer.fit(self.localPoints)

    def _clusters(self, mode):
        clusters = collections.defaultdict(list)
        outliers = []
        for idxPoint, idxCluster in enumerate(self.clusterer.labels_):
            if mode == "localPoints":
                item = self.localPoints[idxPoint]
            elif mode == "indices":
                item = idxPoint
            else:
                raise ValueError()
            if idxCluster >= 0:
                clusters[idxCluster].append(item)
            else:
                outliers.append(item)
        return list(clusters.values()), outliers

    def clustersLocalPoints(self) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        """
        :return: a tuple (clusters, outliers), where clusters is a dictionary mapping from cluster index to
            the list of local points within the cluster and outliers is a list of local points not within
            clusters
        """
        return self._clusters("localPoints")

    def clustersIndices(self) -> Tuple[List[List[int]], List[int]]:
        return self._clusters("indices")


class DBSCANGeoCoordClusterer(SkLearnGeoCoordClusterer):
    def __init__(self, eps, min_samples, lcs: LocalCoordinateSystem = None, **kwargs):
        """
        :param eps: the maximum distance between two samples for one to be considered as in the neighbourhood of the other
        :param min_samples: the minimum number of samples that must be within a neighbourhood for a cluster to be formed
        :param lcs: the local coordinate system for conversion to a Euclidian space
        :param kwargs: additional arguments to pass to DBSCAN (see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
        """
        super().__init__(sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, **kwargs), lcs)