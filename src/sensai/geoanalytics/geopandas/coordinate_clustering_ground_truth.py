"""
This module contains utilities for retrieving and visualizing ground truth labels for evaluating clustering algorithms
"""

import geopandas as gp
import logging
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
from typing import Sequence, Union, Optional

from .coordinates import extractCoordinatesArray, TCoordinates, GeoDataFrameWrapper

log = logging.getLogger(__name__)


class PolygonAnnotatedCoordinates(GeoDataFrameWrapper):
    """
    Class for retrieving ground truth cluster labels from a set of coordinate points and polygons.
    From the provided 2-dim. coordinates only points within the ground truth region will be considered.
    """

    def __init__(self, coordinates: TCoordinates, groundTruthPolygons: Union[str, Sequence[Polygon], GeoDataFrame],
                 noiseLabel: Optional[int] = -1):
        """
        :param coordinates: coordinates of points. These points should be spread over an area larger or equal to
            the ground truth area
        :param groundTruthPolygons: sequence of polygons, GeoDataFrame or path to a shapefile containing such a sequence.
            The polygons represent the ground truth for clustering.
            *Important*: the first polygon in the sequence is assumed to be the region within
            which ground truth was provided and has to cover all remaining polygons. This also means that all non-noise
            clusters in that region should be covered by a polygon
        :param noiseLabel: label to associate with noise or None
        """

        # The constructor might seem bloated but it really mostly does input validation for the polygons
        coordinates = extractCoordinatesArray(coordinates)
        if isinstance(groundTruthPolygons, str):
            polygons: Sequence[Polygon] = gp.read_file(groundTruthPolygons).geometry.values
        elif isinstance(groundTruthPolygons, GeoDataFrame):
            polygons: Sequence[Polygon] = groundTruthPolygons.geometry.values
        else:
            polygons = groundTruthPolygons
        self.regionPolygon = polygons[0]
        self.noiseLabel = noiseLabel
        self.clusterPolygons = MultiPolygon(polygons[1:])
        self.noisePolygon = self.regionPolygon.difference(self.clusterPolygons)

        self.regionMultipoint = MultiPoint(coordinates).intersection(self.regionPolygon)
        if self.regionMultipoint.is_empty:
            raise Exception(f"The ground truth region contains no datapoints. "
                            f"This can happen if you have provided unsuitable coordinates")
        self.noiseMultipoint = self.regionMultipoint.intersection(self.noisePolygon)
        if self.noiseLabel is None and not self.noisePolygon.is_empty:
            raise Exception(f"No noise_label was provided but there is noise: {len(self.noiseMultipoint)} datapoints"
                            f"in annotated area do not belong to any cluster polygon")
        self.clustersMultipoints = []
        intermediatePolygon = Polygon()
        for i, clusterPolygon in enumerate(self.clusterPolygons, start=1):
            if not intermediatePolygon.intersection(clusterPolygon).is_empty:
                raise Exception(f"The polygons should be non-intersecting: polygon {i} intersects with previous polygons")
            intermediatePolygon = intermediatePolygon.union(clusterPolygon)
            clusterMultipoint = self.regionMultipoint.intersection(clusterPolygon)
            if clusterMultipoint.is_empty:
                raise Exception(f"The annotated cluster for polygon {i} is empty - check your data!")
            self.clustersMultipoints.append(clusterMultipoint)

    def toGeoDF(self, crs='epsg:3857', include_noise=True):
        """
        :return: GeoDataFrame with clusters as MultiPoint instance indexed by the clusters' identifiers
        """
        clusters = self.clustersMultipoints
        firstLabel = 0
        if self.noiseLabel is not None and include_noise:
            clusters = [self.noiseMultipoint] + clusters
            firstLabel = self.noiseLabel
        gdf = gp.GeoDataFrame({"geometry": clusters,
                               "identifier": list(range(firstLabel, firstLabel + len(clusters), 1))}, crs=crs)
        gdf.set_index("identifier", drop=True, inplace=True)
        return gdf

    def plot(self, includeNoise=True, **kwargs):
        """
        Plots the ground truth clusters

        :param includeNoise:
        :param kwargs:
        :return:
        """
        gdf = self.toGeoDF(include_noise=includeNoise)
        gdf["color"] = np.random.random(len(gdf))
        if includeNoise and self.noiseLabel is not None:
            gdf.loc[self.noiseLabel, "color"] = 0
        gdf.plot(column="color", **kwargs)

    def getCoordinatesLabels(self):
        """
        Extract cluster coordinates and labels as numpy arrays from the provided ground truth region and
        cluster polygons

        :return: tuple of arrays of the type (coordinates, labels)
        """
        coords, labels = [], []
        for row in self.toGeoDF(include_noise=True).itertuples():
            clusterMultipoint, label = row.geometry, row.Index
            coords += [[p.x, p.y] for p in clusterMultipoint]
            labels += [label] * len(clusterMultipoint)
        return np.array(coords), np.array(labels)
