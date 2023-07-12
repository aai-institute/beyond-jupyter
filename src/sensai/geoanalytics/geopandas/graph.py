import networkx as nx
import numpy as np
import scipy
from itertools import combinations
from scipy.spatial.distance import euclidean
from typing import Callable, Dict
from shapely.geometry import MultiLineString
import geopandas as gp

from .coordinates import extractCoordinatesArray, GeoDataFrameWrapper


def delaunayGraph(data: np.ndarray, edge_weight: Callable[[np.ndarray, np.ndarray], float] = euclidean):
    """
    The Delaunay triangulation of the data as networkx.Graph

    :param data:
    :param edge_weight: function to compute weight given two coordinate points
    :return: instance of networx.Graph where the edges contain additional datapoints entries for
        "weight" and for constants.COORDINATE_PAIR_KEY
    """
    tri = scipy.spatial.Delaunay(data)
    graph = nx.Graph()

    for simplex in tri.simplices:
        for vertex_id_pair in combinations(simplex, 2):
            coordinate_pair = tri.points[
                np.array(vertex_id_pair)]  # vertex_id_pair is a tuple and needs to be cast to an array
            graph.add_edge(*vertex_id_pair, weight=edge_weight(*coordinate_pair))
    return graph


class SpanningTree:
    """
    Wrapper around a tree-finding algorithm that will be applied on the Delaunay graph of the datapoints
    """
    def __init__(self, datapoints: np.ndarray, tree_finder: Callable[[nx.Graph], nx.Graph] = nx.minimum_spanning_tree):
        """
        :param datapoints:
        :param tree_finder: function mapping a graph to a subgraph. The default is minimum_spanning_tree
        """
        datapoints = extractCoordinatesArray(datapoints)
        self.tree = tree_finder(delaunayGraph(datapoints))
        edgeWeights = []
        self.coordinatePairs = []
        for edge in self.tree.edges.data():
            edgeCoordinateIndices, edgeData = [edge[0], edge[1]], edge[2]
            edgeWeights.append(edgeData["weight"])
            self.coordinatePairs.append(datapoints[edgeCoordinateIndices])
        self.edgeWeights = np.array(edgeWeights)

    def totalWeight(self):
        return self.edgeWeights.sum()

    def numEdges(self):
        return len(self.tree.edges)

    def meanEdgeWeight(self):
        return self.edgeWeights.mean()

    def summaryDict(self) -> Dict[str, float]:
        """
        Dictionary containing coarse information about the tree
        """
        return {
            "numEdges": self.numEdges(),
            "totalWeight": self.totalWeight(),
            "meanEdgeWeight": self.meanEdgeWeight()
        }


class CoordinateSpanningTree(SpanningTree, GeoDataFrameWrapper):
    """
    Wrapper around a tree-finding algorithm that will be applied on the Delaunay graph of the coordinates.
    Enhances the :class:`SpanningTree` class by adding methods and validation specific to geospatial coordinates.
    """
    def __init__(self, datapoints: np.ndarray, tree_finder: Callable[[nx.Graph], nx.Graph] = nx.minimum_spanning_tree):
        datapoints = extractCoordinatesArray(datapoints)
        super().__init__(datapoints, tree_finder=tree_finder)

    def multiLineString(self):
        return MultiLineString(self.coordinatePairs)

    def toGeoDF(self, crs='epsg:3857'):
        """
        :param crs: projection. By default pseudo-mercator
        :return: GeoDataFrame of length 1 with the tree as MultiLineString instance
        """
        gdf = gp.GeoDataFrame({"geometry": [self.multiLineString()]})
        gdf.crs = crs
        return gdf