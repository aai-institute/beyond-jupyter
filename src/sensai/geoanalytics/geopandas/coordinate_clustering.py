import logging
from typing import Callable, Union, Iterable

import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint

from .coordinates import validate_coordinates, extract_coordinates_array, TCoordinates, GeoDataFrameWrapper
from ...clustering import SkLearnEuclideanClusterer
from ...clustering.clustering_base import EuclideanClusterer
from ...clustering.sklearn_clustering import SkLearnClustererProtocol
from ...util.cache import LoadSaveInterface
from ...util.profiling import timed

log = logging.getLogger(__name__)


class CoordinateEuclideanClusterer(EuclideanClusterer, GeoDataFrameWrapper):
    """
    Wrapper around a clustering model. This class adds additional, geospatial-specific features to the provided
    clusterer

    :param clusterer: an instance of ClusteringModel
    """
    def __init__(self, clusterer: EuclideanClusterer):
        self.clusterer = clusterer
        super().__init__(noise_label=clusterer.noiseLabel,
            max_cluster_size=clusterer.maxClusterSize, min_cluster_size=clusterer.minClusterSize)

    class Cluster(EuclideanClusterer.Cluster, GeoDataFrameWrapper, LoadSaveInterface):
        """
        Wrapper around a coordinates array

        :param coordinates:
        :param identifier:
        """

        def __init__(self, coordinates: np.ndarray, identifier: Union[str, int]):
            validate_coordinates(coordinates)
            super().__init__(coordinates, identifier)

        def to_geodf(self, crs='epsg:3857'):
            """
            Export the cluster as a GeoDataFrame of length 1 with the cluster as an instance of
            MultiPoint and the identifier as index.

            :param crs: projection. By default pseudo-mercator
            :return: GeoDataFrame
            """
            gdf = gp.GeoDataFrame({"geometry": [self.as_multipoint()]}, index=[self.identifier])
            gdf.index.name = "identifier"
            gdf.crs = crs
            return gdf

        def as_multipoint(self):
            """
            :return: The cluster's coordinates as a MultiPoint object
            """
            return MultiPoint(self.datapoints)

        @classmethod
        def load(cls, path):
            """
            Instantiate from a geopandas readable file containing a single row with an identifier and an instance
            of MultiPoint

            :param path:
            :return: instance of CoordinateCluster
            """
            log.info(f"Loading instance of {cls.__name__} from {path}")
            gdf = gp.read_file(path)
            if len(gdf) != 1:
                raise Exception(f"Expected {path} to contain a single row, instead got {len(gdf)}")
            identifier, multipoint = gdf.identifier.values[0], gdf.geometry.values[0]
            return cls(np.array([[p.x, p.y] for p in multipoint]), identifier)

        def save(self, path, crs="EPSG:3857"):
            """
            Saves the cluster's coordinates as shapefile

            :param crs:
            :param path:
            :return:
            """
            log.info(f"Saving instance of {self.__class__.__name__} as shapefile to {path}")
            self.to_geodf(crs).to_file(path, index=True)

    def _compute_labels(self, x: np.ndarray) -> np.ndarray:
        validate_coordinates(x)
        return self.clusterer._compute_labels(x)

    def fit(self, coordinates: TCoordinates):
        """
        Fitting to coordinates from a numpy array, a MultiPoint object or a GeoDataFrame with one Point per row

        :param coordinates:
        :return:
        """
        coordinates = extract_coordinates_array(coordinates)
        super().fit(coordinates)

    @timed
    def to_geodf(self, condition: Callable[[Cluster], bool] = None, crs='epsg:3857',
            include_noise=False) -> gp.GeoDataFrame:
        """
        GeoDataFrame containing all clusters found by the model.
        It is a concatenation of GeoDataFrames of individual clusters

        :param condition: if provided, only clusters fulfilling the condition will be included
        :param crs: projection. By default pseudo-mercator
        :param include_noise:
        :return: GeoDataFrame with all clusters indexed by their identifier
        """
        geodf = gp.GeoDataFrame()
        geodf.crs = crs
        for cluster in self.clusters(condition):
            geodf = pd.concat((geodf, cluster.to_geodf(crs=crs)))
        if include_noise:
            geodf = pd.concat((geodf, self.noise_cluster().to_geodf(crs=crs)))
        return geodf

    def plot(self, include_noise=False, condition=None, **kwargs):
        """
        Plots the resulting clusters with random coloring

        :param include_noise: Whether to include the noise cluster
        :param condition: If provided, only clusters fulfilling this condition will be included
        :param kwargs: passed to GeoDataFrame.plot
        :return:
        """
        geodf = self.to_geodf(condition=condition, include_noise=include_noise)
        geodf["color"] = np.random.random(len(geodf))
        if include_noise:
            geodf.loc[self.noiseLabel, "color"] = 0
        geodf.plot(column="color", **kwargs)

    # the overriding of the following methods is only necessary for getting the type annotations right
    # if mypy ever permits annotating nested classes correctly, these methods can be removed
    def get_cluster(self, cluster_id: int) -> Cluster:
        return super().get_cluster(cluster_id)

    def noise_cluster(self) -> Cluster:
        return super().noise_cluster()

    def clusters(self, condition: Callable[[Cluster], bool] = None) -> Iterable[Cluster]:
        return super().clusters(condition=condition)


class SkLearnCoordinateClustering(CoordinateEuclideanClusterer):
    """
    Wrapper around a sklearn clusterer. This class adds additional features like relabelling and convenient methods
    for handling geospatial data

    :param clusterer: a clusterer object compatible the sklearn API
    :param noise_label: label that is associated with the noise cluster or None
    :param min_cluster_size: if not None, clusters below this size will be labeled as noise
    :param max_cluster_size: if not None, clusters above this size will be labeled as noise
    """
    def __init__(self, clusterer: SkLearnClustererProtocol, noise_label=-1,
                 min_cluster_size: int = None, max_cluster_size: int = None):
        clusterer = SkLearnEuclideanClusterer(clusterer, noise_label=noise_label,
                           min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size)
        super().__init__(clusterer)
