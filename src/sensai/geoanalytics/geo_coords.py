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


def longitudePerM(latitude):
    return LATITUDE_PER_METRE / math.cos(math.radians(latitude))


def latitudePerM():
    return LATITUDE_PER_METRE


def approximateSquaredDistance(p1: Tuple[float, float], p2: Tuple[float, float]):
    """
    :param p1: a tuple (latitude, longitude)
    :param p2: a tuple (latitude, longitude)
    :return: the approximate squared distance (in m²) between p1 and p2
    """
    latPerM = latitudePerM()
    p1lat, p1lon = p1
    p2lat, p2lon = p2
    lonPerM = longitudePerM((p1lat + p2lat) / 2)
    dx = (p2lon - p1lon) / lonPerM
    dy = (p2lat - p1lat) / latPerM
    return dx * dx + dy * dy


def closestPointOnSegment(searchPos: Tuple[float, float], segPoint1: Tuple[float, float], segPoint2: Tuple[float, float]):
    """
    Gets the point on the line segment connecting segPoint1 and segPoint2 that is closest to searchPos

    :param searchPos: the position for which to search for the closest point on the line segment
    :param segPoint1: the first point defining the line segment on which to search
    :param segPoint2: the second point defining the line segment on which to search
    :return: the closest point, which is on the line connecting segPoint1 and segPoint2 (and may be one of the two points)
    """
    seg1lat, seg1lon = segPoint1
    seg2lat, seg2lon = segPoint2
    srchlat, srchlon = searchPos
    latPerM = latitudePerM()
    lonPerM = longitudePerM(srchlat)
    sp1x = (seg1lon - srchlon) / lonPerM
    sp1y = (seg1lat - srchlat) / latPerM
    sp2x = (seg2lon - srchlon) / lonPerM
    sp2y = (seg2lat - srchlat) / latPerM
    vx = sp2x - sp1x
    vy = sp2y - sp1y
    c1 = -vx * sp1x - vy * sp1y
    if c1 <= 0:
        return segPoint1
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return segPoint2
    b = 0 if c2 == 0 else c1 / c2
    lon = seg1lon + b * vx * lonPerM
    lat = seg1lat + b * vy * latPerM
    return [lat, lon]


def orientation(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Gets the orientation angle for the vector from p1 to p2

    :param p1: a (lat, lon) pair
    :param p2: a (lat, lon) pair
    :return: the orientation angle in rad
    """
    p1Lat, p1Lon = p1
    p2Lat, p2Lon = p2
    centerLat = (p1Lat + p2Lat) / 2
    dx = (p2Lon - p1Lon) / longitudePerM(centerLat)
    dy = (p2Lat - p1Lat) / latitudePerM()
    return math.atan2(dy, dx)


def absAngleDifference(a1: float, a2: float) -> float:
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


def closestPointOnPolyline(searchPos, polyline, searchOrientationAngle=None, maxAngleDifference=0) -> Tuple[Tuple[float, float], float, int]:
    """
    Gets the point on the given polyline that is closest to the given search position along with the
    distance (in metres) to the polyline

    :param searchPos: a (lat, lon) pair indicating the position for which to find the closest math on the polyline
    :param polyline: list of (lat, lon) pairs that make up the polyline on which to search
    :param searchOrientationAngle: if not None, defines the orientation with which to compute angle differences (if maxAngleDifference > 0)
    :param maxAngleDifference: the maximum absolute angle difference (in rad) that is admissible (between the orientation of the
        respective line segment and the orientation given in searchOrientationAngle)
    :return: a tuple (optPoint, optDist, optSegmentStartIdx) where
        optPoint is the closest point (with admissible orientation - or None if there is none),
        optDist is the distance from the polyline to the closest point,
        optSegmentStartIdx is the index of the first point of the segment on the polyline for which the closest point was found
    """
    if len(polyline) < 2:
        raise Exception("Polyline must consist of at least two points")
    optSegmentStartIdx = None
    optPoint = None
    optSqDist = None
    for i in range(len(polyline)-1):
        if maxAngleDifference > 0:
            orientationAngle = orientation(polyline[i], polyline[i+1])
            angDiff = absAngleDifference(orientationAngle, searchOrientationAngle)
            if angDiff > maxAngleDifference:
                continue
        optSegPoint = closestPointOnSegment(searchPos, polyline[i], polyline[i + 1])
        sqDist = approximateSquaredDistance(searchPos, optSegPoint)
        if optSqDist is None or sqDist < optSqDist:
            optPoint = optSegPoint
            optSqDist = sqDist
            optSegmentStartIdx = i
    return optPoint, math.sqrt(optSqDist), optSegmentStartIdx


class GeoCoord(ToStringMixin):
    """
    Represents geographic coordinates (WGS84)
    """
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def latlon(self):
        return self.lat, self.lon

    def distanceTo(self, gpsPosition: 'GeoCoord'):
        return math.sqrt(self.squaredDistanceTo(gpsPosition))

    def squaredDistanceTo(self, gpsPosition: 'GeoCoord'):
        return approximateSquaredDistance(self.latlon(), gpsPosition.latlon())

    def localCoords(self, lcs):
        return lcs.getLocalCoords(self.lat, self.lon)

    @classmethod
    def meanCoord(cls, geoCoords: Iterable["GeoCoord"]):
        meanLat = np.mean([c.lat for c in geoCoords])
        meanLon = np.mean([c.lon for c in geoCoords])
        # noinspection PyTypeChecker
        return GeoCoord(meanLat, meanLon)


class GpsTracePoint(GeoCoord):
    def __init__(self, lat, lon, time: pd.Timestamp):
        super().__init__(lat, lon)
        self.time = time


class GeoRect:
    def __init__(self, minLat: float, minLon: float, maxLat: float, maxLon: float):
        if maxLat < minLat or maxLon < minLon:
            raise ValueError()
        self.minLat = minLat
        self.minLon = minLon
        self.maxLat = maxLat
        self.maxLon = maxLon

    @staticmethod
    def fromCircle(centreLat, centreLon, radiusM):
        """Creates the bounding rectangle for the given circular area"""
        from .local_coords import LocalCoordinateSystem
        lcs = LocalCoordinateSystem(centreLat, centreLon)
        minLat, minLon = lcs.getLatLon(-radiusM, -radiusM)
        maxLat, maxLon = lcs.getLatLon(radiusM, radiusM)
        return GeoRect(minLat, minLon, maxLat, maxLon)
