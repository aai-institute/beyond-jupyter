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
    def fit_geo_coords(self, geo_coords: List[GeoCoord]):
        """
        :param geo_coords: the coordinates to be clustered
        """
        pass

    @abstractmethod
    def clusters_indices(self) -> Tuple[List[List[int]], List[int]]:
        """
        :return: a tuple (clusters, outliers), where clusters is a list of point indices, one list for each
            cluster containing the indices of points within the cluster, and outliers is the list of indices of points not within
            clusters
        """
        pass


class GreedyAgglomerativeGeoCoordClusterer(GeoCoordClusterer):
    def __init__(self,
            max_min_distance_for_merge_m: float,
            max_distance_m: float,
            min_cluster_size: int,
            lcs: LocalCoordinateSystem = None):
        """
        :param max_min_distance_for_merge_m: the maximum distance, in metres, for the minimum distance between two existing clusters for a merge
            to be admissible
        :param max_distance_m: the maximum distance, in metres, between any two points for the points to be allowed to be in the same cluster
        :param min_cluster_size: the minimum number of points any valid cluster must ultimately contain; the points in any smaller clusters
            shall be considered as outliers
        :param lcs: the local coordinate system to use for clustering; if None, compute based on mean coordinates passed when fitting
        """
        self.lcs = lcs
        self.min_cluster_size = min_cluster_size
        self.max_min_distance_for_merge = max_min_distance_for_merge_m
        self.max_distance_m = max_distance_m
        self.squared_max_min_distance_for_merge = max_min_distance_for_merge_m * max_min_distance_for_merge_m
        self.squared_max_distance = max_distance_m * max_distance_m
        self.local_points = None
        self.m_min_distance: Optional["GreedyAgglomerativeGeoCoordClusterer.Matrix"] = None
        self.m_max_squared_distance: Optional["GreedyAgglomerativeGeoCoordClusterer.Matrix"] = None
        self.clusters = None

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
        def __init__(self, point: "GreedyAgglomerativeGeoCoordClusterer.LocalPoint", idx: int,
                clusterer: 'GreedyAgglomerativeGeoCoordClusterer'):
            self.idx = idx
            self.clusterer = clusterer
            self.points = [point]

        def merge_cost(self, other: "GreedyAgglomerativeGeoCoordClusterer.Cluster"):
            cartesian_product = itertools.product(self.points, other.points)
            min_squared_distance = math.inf
            max_squared_distance = 0
            for p1, p2 in cartesian_product:
                diff = p1.xy - p2.xy
                squared_distance = np.dot(diff, diff)
                if squared_distance > self.clusterer.squared_max_distance:
                    max_squared_distance = math.inf
                    break
                else:
                    min_squared_distance = min(squared_distance, min_squared_distance)

            # fill cache: the max value takes precedence; if it is inf (no merge admissible), then the min value is also set to inf;
            # the min value valid only if the max value is finite
            self.clusterer.m_max_squared_distance.set(self.idx, other.idx, max_squared_distance)
            if np.isinf(max_squared_distance):
                self.clusterer.m_min_distance.set(self.idx, other.idx, np.inf)
            else:
                self.clusterer.m_min_distance.set(self.idx, other.idx, min_squared_distance)

            if np.isinf(max_squared_distance):
                return math.inf
            if min_squared_distance <= self.clusterer.squared_max_min_distance_for_merge:
                return min_squared_distance
            return math.inf

        def merge(self, other):
            self.points += other.points

    def fit_geo_coords(self, geo_coords: List[GeoCoord]) -> None:
        self.m_min_distance = self.Matrix(len(geo_coords))
        self.m_max_squared_distance = self.Matrix(len(geo_coords))
        if self.lcs is None:
            mean_coord = GeoCoord.mean_coord(geo_coords)
            self.lcs = LocalCoordinateSystem(mean_coord.lat, mean_coord.lon)
        self.local_points = [self.LocalPoint(np.array(self.lcs.get_local_coords(p.lat, p.lon)), idx) for idx, p in enumerate(geo_coords)]
        clusters = [self.Cluster(lp, i, self) for i, lp in enumerate(self.local_points)]
        gac = GreedyAgglomerativeClustering(clusters,
            merge_candidate_determination_strategy=self.MergeCandidateDeterminationStrategy(self.max_distance_m, self))
        clusters = gac.apply_clustering()
        self.clusters = clusters

    def clusters_indices(self) -> Tuple[List[List[int]], List[int]]:
        outliers = []
        clusters = []
        for c in self.clusters:
            indices = [p.idx for p in c.points]
            if len(c.points) < self.min_cluster_size:
                outliers.extend(indices)
            else:
                clusters.append(indices)
        return clusters, outliers

    class MergeCandidateDeterminationStrategy(GreedyAgglomerativeClustering.MergeCandidateDeterminationStrategy):
        def __init__(self, search_radius_m: float, parent: "GreedyAgglomerativeGeoCoordClusterer"):
            super().__init__()
            self.parent = parent
            self.searchRadiusM = search_radius_m

        def set_clusterer(self, clusterer: GreedyAgglomerativeClustering):
            super().set_clusterer(clusterer)
            points = []
            for wc in self.clusterer.wrapped_clusters:
                c: GreedyAgglomerativeGeoCoordClusterer.Cluster = wc.cluster
                for p in c.points:
                    points.append(p.xy)
            assert len(points) == len(self.clusterer.wrapped_clusters)
            points = np.stack(points)
            self.kdtree = sklearn.neighbors.KDTree(points)

        def iter_candidate_indices(self, wc: "GreedyAgglomerativeClustering.WrappedCluster", initial: bool,
                merged_cluster_indices: Tuple[int, int] = None) -> Iterator[int]:
            c: GreedyAgglomerativeGeoCoordClusterer.Cluster = wc.cluster
            if initial:
                local_point = c.points[0]  # pick any point from wc, since we use maximum cluster extension as search radius
                indices = self.kdtree.query_radius(np.reshape(local_point.xy, (1, 2)), self.searchRadiusM)[0]
                candidate_set = set()
                for idx in indices:
                    wc = self.clusterer.wrapped_clusters[idx]
                    candidate_set.add(wc.get_cluster_association().idx)
                yield from sorted(candidate_set)
            else:
                # The new distance values (max/min) between wc and any cluster index otherIdx can be computed from the cached distance
                # values of the two clusters from which wc was created through a merge:
                # The max distance is the maximum of the squared distances of the original clusters (and if either is inf, then
                # a merge is definitely inadmissible, because one of the original clusters was already too far away).
                # The min distance is the minimum of the squred distances of the original clusters.
                c1, c2 = merged_cluster_indices
                max1 = self.parent.m_max_squared_distance.m[c1]
                max2 = self.parent.m_max_squared_distance.m[c2]
                max_combined = np.maximum(max1, max2)
                for otherIdx, maxSqDistance in enumerate(max_combined):
                    min_sq_distance = np.inf
                    if maxSqDistance <= self.parent.squared_max_distance:
                        wc_other = self.clusterer.wrapped_clusters[otherIdx]
                        if wc_other.is_merged():
                            continue
                        min1 = self.parent.m_min_distance.get(c1, otherIdx)
                        min2 = self.parent.m_min_distance.get(c2, otherIdx)
                        min_sq_distance = min(min1, min2)
                        if min_sq_distance <= self.parent.squared_max_min_distance_for_merge:
                            yield GreedyAgglomerativeClustering.ClusterMerge(wc, wc_other, min_sq_distance)
                    # update cache
                    self.parent.m_max_squared_distance.set(wc.idx, otherIdx, maxSqDistance)
                    self.parent.m_min_distance.set(wc.idx, otherIdx, min_sq_distance)


class SkLearnGeoCoordClusterer(GeoCoordClusterer):
    def __init__(self, clusterer, lcs: LocalCoordinateSystem = None):
        """
        :param clusterer: a clusterer from sklearn.cluster
        :param lcs: the local coordinate system to use for Euclidian conversion; if None, determine from data (using mean coordinate as
            centre)
        """
        self.lcs = lcs
        self.clusterer = clusterer
        self.local_points = None

    def fit_geo_coords(self, geo_coords: List[GeoCoord]):
        if self.lcs is None:
            mean_coord = GeoCoord.mean_coord(geo_coords)
            self.lcs = LocalCoordinateSystem(mean_coord.lat, mean_coord.lon)
        self.local_points = [self.lcs.get_local_coords(p.lat, p.lon) for p in geo_coords]
        self.clusterer.fit(self.local_points)

    def _clusters(self, mode):
        clusters = collections.defaultdict(list)
        outliers = []
        for idxPoint, idxCluster in enumerate(self.clusterer.labels_):
            if mode == "localPoints":
                item = self.local_points[idxPoint]
            elif mode == "indices":
                item = idxPoint
            else:
                raise ValueError()
            if idxCluster >= 0:
                clusters[idxCluster].append(item)
            else:
                outliers.append(item)
        return list(clusters.values()), outliers

    def clusters_local_points(self) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        """
        :return: a tuple (clusters, outliers), where clusters is a dictionary mapping from cluster index to
            the list of local points within the cluster and outliers is a list of local points not within
            clusters
        """
        return self._clusters("localPoints")

    def clusters_indices(self) -> Tuple[List[List[int]], List[int]]:
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
