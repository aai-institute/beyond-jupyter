"""
Utility functions and classes for geographic coordinates
"""

import math
from typing import Tuple, Iterable

import numpy as np
import pandas as pd

from ..util.string import ToStringMixin

EARTH_RADIUS = 6371000
EARTH_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS
LATITUDE_PER_METRE = 360.0 / EARTH_CIRCUMFERENCE


def longitude_per_m(latitude):
    return LATITUDE_PER_METRE / math.cos(math.radians(latitude))


def latitude_per_m():
    return LATITUDE_PER_METRE


def approximate_squared_distance(p1: Tuple[float, float], p2: Tuple[float, float]):
    """
    :param p1: a tuple (latitude, longitude)
    :param p2: a tuple (latitude, longitude)
    :return: the approximate squared distance (in m²) between p1 and p2
    """
    lat_per_m = latitude_per_m()
    p1lat, p1lon = p1
    p2lat, p2lon = p2
    lon_per_m = longitude_per_m((p1lat + p2lat) / 2)
    dx = (p2lon - p1lon) / lon_per_m
    dy = (p2lat - p1lat) / lat_per_m
    return dx * dx + dy * dy


def closest_point_on_segment(search_pos: Tuple[float, float], segPoint1: Tuple[float, float], segPoint2: Tuple[float, float]):
    """
    Gets the point on the line segment connecting segPoint1 and segPoint2 that is closest to searchPos

    :param search_pos: the position for which to search for the closest point on the line segment
    :param segPoint1: the first point defining the line segment on which to search
    :param segPoint2: the second point defining the line segment on which to search
    :return: the closest point, which is on the line connecting segPoint1 and segPoint2 (and may be one of the two points)
    """
    seg1lat, seg1lon = segPoint1
    seg2lat, seg2lon = segPoint2
    srchlat, srchlon = search_pos
    lat_per_m = latitude_per_m()
    lon_per_m = longitude_per_m(srchlat)
    sp1x = (seg1lon - srchlon) / lon_per_m
    sp1y = (seg1lat - srchlat) / lat_per_m
    sp2x = (seg2lon - srchlon) / lon_per_m
    sp2y = (seg2lat - srchlat) / lat_per_m
    vx = sp2x - sp1x
    vy = sp2y - sp1y
    c1 = -vx * sp1x - vy * sp1y
    if c1 <= 0:
        return segPoint1
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return segPoint2
    b = 0 if c2 == 0 else c1 / c2
    lon = seg1lon + b * vx * lon_per_m
    lat = seg1lat + b * vy * lat_per_m
    return [lat, lon]


def orientation(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Gets the orientation angle for the vector from p1 to p2

    :param p1: a (lat, lon) pair
    :param p2: a (lat, lon) pair
    :return: the orientation angle in rad
    """
    p1_lat, p1_lon = p1
    p2_lat, p2_lon = p2
    center_lat = (p1_lat + p2_lat) / 2
    dx = (p2_lon - p1_lon) / longitude_per_m(center_lat)
    dy = (p2_lat - p1_lat) / latitude_per_m()
    return math.atan2(dy, dx)


def abs_angle_difference(a1: float, a2: float) -> float:
    """
    Computes the absolute angle difference in ]-pi, pi] between two angles

    :param a1: an angle in rad
    :param a2: an angle in rad
    :return: the difference in rad
    """
    d = a1 - a2
    while d > math.pi:
        d -= 2*math.pi
    while d <= -math.pi:
        d += 2*math.pi
    return abs(d)


def closest_point_on_polyline(search_pos, polyline, search_orientation_angle=None, max_angle_difference=0) \
        -> Tuple[Tuple[float, float], float, int]:
    """
    Gets the point on the given polyline that is closest to the given search position along with the
    distance (in metres) to the polyline

    :param search_pos: a (lat, lon) pair indicating the position for which to find the closest math on the polyline
    :param polyline: list of (lat, lon) pairs that make up the polyline on which to search
    :param search_orientation_angle: if not None, defines the orientation with which to compute angle differences
        (if maxAngleDifference > 0)
    :param max_angle_difference: the maximum absolute angle difference (in rad) that is admissible (between the orientation of the
        respective line segment and the orientation given in searchOrientationAngle)
    :return: a tuple (opt_point, opt_dist, opt_segment_start_idx) where
        opt_point is the closest point (with admissible orientation - or None if there is none),
        opt_dist is the distance from the polyline to the closest point,
        opt_segment_start_idx is the index of the first point of the segment on the polyline for which the closest point was found
    """
    if len(polyline) < 2:
        raise Exception("Polyline must consist of at least two points")
    opt_segment_start_idx = None
    opt_point = None
    opt_sq_dist = None
    for i in range(len(polyline)-1):
        if max_angle_difference > 0:
            orientation_angle = orientation(polyline[i], polyline[i+1])
            ang_diff = abs_angle_difference(orientation_angle, search_orientation_angle)
            if ang_diff > max_angle_difference:
                continue
        opt_seg_point = closest_point_on_segment(search_pos, polyline[i], polyline[i + 1])
        sq_dist = approximate_squared_distance(search_pos, opt_seg_point)
        if opt_sq_dist is None or sq_dist < opt_sq_dist:
            opt_point = opt_seg_point
            opt_sq_dist = sq_dist
            opt_segment_start_idx = i
    return opt_point, math.sqrt(opt_sq_dist), opt_segment_start_idx


class GeoCoord(ToStringMixin):
    """
    Represents geographic coordinates (WGS84)
    """
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def latlon(self):
        return self.lat, self.lon

    def distance_to(self, gps_position: 'GeoCoord'):
        return math.sqrt(self.squared_distance_to(gps_position))

    def squared_distance_to(self, gps_position: 'GeoCoord'):
        return approximate_squared_distance(self.latlon(), gps_position.latlon())

    def local_coords(self, lcs):
        return lcs.get_local_coords(self.lat, self.lon)

    @classmethod
    def mean_coord(cls, geo_coords: Iterable["GeoCoord"]):
        mean_lat = np.mean([c.lat for c in geo_coords])
        mean_lon = np.mean([c.lon for c in geo_coords])
        # noinspection PyTypeChecker
        return GeoCoord(mean_lat, mean_lon)


class GpsTracePoint(GeoCoord):
    def __init__(self, lat, lon, time: pd.Timestamp):
        super().__init__(lat, lon)
        self.time = time


class GeoRect:
    def __init__(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float):
        if max_lat < min_lat or max_lon < min_lon:
            raise ValueError()
        self.minLat = min_lat
        self.minLon = min_lon
        self.maxLat = max_lat
        self.maxLon = max_lon

    @staticmethod
    def from_circle(centre_lat, centre_lon, radius_m):
        """Creates the bounding rectangle for the given circular area"""
        from .local_coords import LocalCoordinateSystem
        lcs = LocalCoordinateSystem(centre_lat, centre_lon)
        min_lat, min_lon = lcs.get_lat_lon(-radius_m, -radius_m)
        max_lat, max_lon = lcs.get_lat_lon(radius_m, radius_m)
        return GeoRect(min_lat, min_lon, max_lat, max_lon)
