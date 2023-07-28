# -*- coding: utf-8 -*-
import logging
import math
import queue
from abc import ABC, abstractmethod
from typing import List, Sequence, Iterator, Callable, Optional, Union, Tuple

log = logging.getLogger(__name__)


class GreedyAgglomerativeClustering(object):
    """
    An implementation of greedy agglomerative clustering which avoids unnecessary 
    recomputations of merge costs through the management of a priority queue of 
    potential merges.

    Greedy agglomerative clustering works as follows. Starting with an initial
    set of clusters (where each cluster typically contains a single data point),
    the method successively merges the two clusters where the merge cost is lowest (greedy),
    until no further merges are admissible.
    The merge operation is a mutating operation, i.e. the initial clusters are modified.

    To apply the method, the Cluster class must be subclassed, so as to define
    what the cost of a merge in your application shall be and how two clusters can be merged.
    For example, if data points are points in a Cartesian coordinate system, then the merge cost
    can be defined as the minimum or maximum distance among all pairs of points in the two clusters,
    admissibility being determined by a threshold that must not be exceeded;
    the merge operation can simply concatenate lists of data points.
    """
    log = log.getChild(__qualname__)

    class Cluster(ABC):
        """
        Base class for clusters that can be merged via GreedyAgglomerativeClustering
        """
        @abstractmethod
        def merge_cost(self, other) -> float:
            """
            Computes the cost of merging the given cluster with this cluster

            :return: the (non-negative) merge cost or math.inf if a merge is inadmissible"""
            pass

        @abstractmethod
        def merge(self, other):
            """
            Merges the given cluster into this cluster"

            :param other: the cluster that is to be merged into this cluster
            """
            pass

    def __init__(self, clusters: Sequence[Cluster],
            merge_candidate_determination_strategy: "GreedyAgglomerativeClustering.MergeCandidateDeterminationStrategy" = None):
        """
        :param clusters: the initial clusters, which are to be agglomerated into larger clusters
        """
        self.prioritised_merges = queue.PriorityQueue()
        self.wrapped_clusters = []
        for idx, c in enumerate(clusters):
            self.wrapped_clusters.append(GreedyAgglomerativeClustering.WrappedCluster(c, idx, self))

        # initialise merge candidate determination strategy
        if merge_candidate_determination_strategy is None:
            merge_candidate_determination_strategy = self.MergeCandidateDeterminationStrategyDefault()
        merge_candidate_determination_strategy.set_clusterer(self)
        self.mergeCandidateDeterminationStrategy = merge_candidate_determination_strategy
        
    def apply_clustering(self) -> List[Cluster]:
        """
        Applies greedy agglomerative clustering to the clusters given at construction, merging
        clusters until no further merges are admissible

        :return: the list of agglomerated clusters (subset of the original clusters, which may have had other
            clusters merged into them)
        """
        # compute all possible merges, adding them to the priority queue
        self.log.debug("Computing initial merges")
        for idx, wc in enumerate(self.wrapped_clusters):
            self.log.debug("Computing potential merges for cluster index %d" % idx)
            wc.compute_merges(True)
        
        # greedily apply the least-cost merges
        steps = 0
        while not self.prioritised_merges.empty():
            self.log.debug("Clustering step %d" % (steps+1))
            have_merge = False
            while not have_merge and not self.prioritised_merges.empty():
                merge = self.prioritised_merges.get()
                if not merge.evaporated:
                    have_merge = True
            if have_merge:
                merge.apply()
            steps += 1
        
        result = filter(lambda wc: not wc.is_merged(), self.wrapped_clusters)
        result = list(map(lambda wc: wc.cluster, result))
        return result

    class WrappedCluster(object):
        """
        Wrapper for clusters which stores additional data required for clustering (internal use only)
        """
        def __init__(self, cluster, idx, clusterer: "GreedyAgglomerativeClustering"):
            self.merged_into_cluster: Optional[GreedyAgglomerativeClustering.WrappedCluster] = None
            self.merges = []
            self.cluster = cluster
            self.idx = idx
            self.clusterer = clusterer

        def is_merged(self) -> bool:
            return self.merged_into_cluster is not None

        def get_cluster_association(self) -> "GreedyAgglomerativeClustering.WrappedCluster":
            """
            Gets the wrapped cluster that this cluster's points have ultimately been merged into (which may be the cluster itself)

            :return: the wrapped cluster this cluster's points are associated with
            """
            if self.merged_into_cluster is None:
                return self
            else:
                return self.merged_into_cluster.get_cluster_association()
            
        def remove_merges(self):
            for merge in self.merges:
                merge.evaporated = True
            self.merges = []

        def compute_merges(self, initial: bool, merged_cluster_indices: Tuple[int, int] = None):
            # add new merges to queue
            wrapped_clusters = self.clusterer.wrapped_clusters
            for item in self.clusterer.mergeCandidateDeterminationStrategy.iter_candidate_indices(self, initial, merged_cluster_indices):
                merge: Optional[GreedyAgglomerativeClustering.ClusterMerge] = None
                if type(item) == int:
                    other_idx = item
                    if other_idx != self.idx:
                        other = wrapped_clusters[other_idx]
                        if not other.is_merged():
                            merge_cost = self.cluster.merge_cost(other.cluster)
                            if not math.isinf(merge_cost):
                                merge = GreedyAgglomerativeClustering.ClusterMerge(self, other, merge_cost)
                else:
                    merge = item
                    assert merge.c1.idx == self.idx
                if merge is not None:
                    merge.c1.merges.append(merge)
                    merge.c2.merges.append(merge)
                    self.clusterer.prioritised_merges.put(merge)

        def __str__(self):
            return "Cluster[idx=%d]" % self.idx
        
    class ClusterMerge(object):
        """
        Represents a potential merge
        """
        log = log.getChild(__qualname__)

        def __init__(self, c1: "GreedyAgglomerativeClustering.WrappedCluster", c2: "GreedyAgglomerativeClustering.WrappedCluster",
                merge_cost):
            self.c1 = c1
            self.c2 = c2
            self.merge_cost = merge_cost
            self.evaporated = False

        def apply(self):
            c1, c2 = self.c1, self.c2
            self.log.debug("Merging %s into %s..." % (str(c1), str(c2)))
            c1.cluster.merge(c2.cluster)
            c2.merged_into_cluster = c1
            c1.remove_merges()
            c2.remove_merges()
            self.log.debug("Computing new merge costs for %s..." % str(c1))
            c1.compute_merges(False, merged_cluster_indices=(c1.idx, c2.idx))
        
        def __lt__(self, other):
            return self.merge_cost < other.merge_cost

    class MergeCandidateDeterminationStrategy(ABC):
        def __init__(self):
            self.clusterer: Optional["GreedyAgglomerativeClustering"] = None

        """
        Determines the indices of clusters which should be evaluated with regard to their merge costs
        """
        def set_clusterer(self, clusterer: "GreedyAgglomerativeClustering"):
            """
            Initialises the clusterer the strategy is applied to
            :param clusterer: the clusterer
            """
            self.clusterer = clusterer

        @abstractmethod
        def iter_candidate_indices(self, wc: "GreedyAgglomerativeClustering.WrappedCluster", initial: bool,
                merged_cluster_indices: Tuple[int, int] = None) -> Iterator[Union[int, "GreedyAgglomerativeClustering.ClusterMerge"]]:
            """
            :param wc: the wrapped cluster: the cluster for which to determine the cluster indices that are to be considered for
                a potential merge
            :param initial: whether we are computing the initial candidates (at the start of the clustering algorithm)
            :param merged_cluster_indices: [for initial=False] the pair of cluster indices that were just joined to form the updated
                cluster wc
            :return: an iterator of cluster indices that should be evaluated as potential merge partners for wc (it may contain the
                index of wc, which will be ignored)
            """
            pass

    class MergeCandidateDeterminationStrategyDefault(MergeCandidateDeterminationStrategy):
        def iter_candidate_indices(self, wc: "GreedyAgglomerativeClustering.WrappedCluster", initial: bool,
                merged_cluster_indices: Tuple[int, int] = None) -> Iterator[Union[int, "GreedyAgglomerativeClustering.ClusterMerge"]]:
            n = len(self.clusterer.wrapped_clusters)
            if initial:
                return range(wc.idx + 1, n)
            else:
                return range(n)
