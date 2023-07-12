import logging
import numpy as np
from itertools import combinations
from scipy.spatial.distance import euclidean
from scipy.spatial.qhull import Delaunay
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import polygonize, unary_union

from .coordinates import extractCoordinatesArray, TCoordinates
from .graph import delaunayGraph

log = logging.getLogger(__name__)


# after already having implemented this, I found the following package: https://github.com/bellockk/alphashape
# It will compute the same polygons (I have verified it). It also contains an optimizer for alpha, which is, however,
# extremely slow and therefore unusable in most practical applications.
def alphaShape(coordinates: TCoordinates, alpha=0.5):
    """
    Compute the `alpha shape`_ of a set of points. Based on `this implementation`_. In contrast to the standard
    definition of the parameter alpha here we normalize it by the mean edge size of the cluster. This results in
    similar "concavity properties" of the resulting shapes for different coordinate sets and a fixed alpha.

    .. _this implementation: https://sgillies.net/2012/10/13/the-fading-shape-of-alpha.html
    .. _alpha shape: https://en.wikipedia.org/wiki/Alpha_shape

    :param coordinates: a suitable iterable of 2-dimensional coordinates
    :param alpha: alpha value to influence the gooeyness of the border. Larger numbers
        don't fall inward as much as smaller numbers.
    :return: a shapely Polygon
    """
    coordinates = extractCoordinatesArray(coordinates)

    edge_index_pairs = set()
    edge_vertex_pairs = []
    graph = delaunayGraph(coordinates)
    mean_edge_size = graph.size(weight="weight") / graph.number_of_edges()

    def add_edge(edge_index_pair, edge_vertex_pair):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        edge_index_pair = tuple(sorted(edge_index_pair))
        if edge_index_pair in edge_index_pairs:
            # already added
            return
        edge_index_pairs.add(edge_index_pair)
        edge_vertex_pairs.append(edge_vertex_pair)

    tri = Delaunay(coordinates)
    for simplex in tri.simplices:
        vertices = tri.points[simplex]
        area = Polygon(vertices).area
        edges = combinations(vertices, 2)
        product_edges_lengths = 1
        for vertex_1, vertex_2 in edges:
            product_edges_lengths *= euclidean(vertex_1, vertex_2)
        # this is the radius of the circumscribed circle of the triangle
        # see https://en.wikipedia.org/wiki/Circumscribed_circle#Triangles
        circum_r = product_edges_lengths / (4.0 * area)

        if circum_r < mean_edge_size/alpha:
            for index_pair in combinations(simplex, 2):
                add_edge(index_pair, tri.points[np.array(index_pair)])

    remaining_edges = MultiLineString(edge_vertex_pairs)

    return unary_union(list(polygonize(remaining_edges)))
