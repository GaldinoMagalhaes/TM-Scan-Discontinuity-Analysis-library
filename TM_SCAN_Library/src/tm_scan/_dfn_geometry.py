"""Exact common trace of a planar disk and a rectangular sampling window."""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _rectangle_trace_length(center, normal, radius, frame, tolerance):
    u, v = frame[0], frame[1]
    hu, hv = frame[2, 0], frame[2, 1]
    a, b = np.dot(normal, u), np.dot(normal, v)
    squared = a*a+b*b
    if squared <= tolerance*tolerance:
        return 0.
    distance = np.dot(normal, center)
    origin = (distance/squared)*(a*u+b*v)
    direction = (-b*u+a*v)/math.sqrt(squared)
    relative = center-origin
    middle = np.dot(relative, direction)
    perpendicular_sq = max(0., np.dot(relative, relative)-middle*middle)
    if math.sqrt(perpendicular_sq) > radius+tolerance:
        return 0.
    half_chord = math.sqrt(max(0., radius*radius-perpendicular_sq))
    lower, upper = middle-half_chord, middle+half_chord
    for axis, half in ((u, hu), (v, hv)):
        position, slope = np.dot(origin, axis), np.dot(direction, axis)
        if abs(slope) <= tolerance:
            if abs(position) > half+tolerance:
                return 0.
        else:
            first, last = (-half-position)/slope, (half-position)/slope
            lower = max(lower, min(first, last))
            upper = min(upper, max(first, last))
        if upper <= lower:
            return 0.
    length = upper-lower
    return length if length > tolerance else 0.


@njit(cache=True, nogil=True)
def _rectangle_trace_lengths_batch(centers, normals, radii, frame, tolerance):
    output = np.empty(len(radii))
    for i in range(len(radii)):
        output[i] = _rectangle_trace_length(centers[i], normals[i], radii[i], frame, tolerance)
    return output
