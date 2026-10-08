from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from manimlib.constants import DEG, RADIANS
from manimlib.constants import FRAME_SHAPE
from manimlib.constants import DOWN, LEFT, ORIGIN, OUT, RIGHT, UP
from manimlib.constants import PI, TAU
from manimlib.mobject.mobject import Mobject
from manimlib.utils.space_ops import normalize
from manimlib.utils.simple_functions import clip

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import Vect3


class CameraFrame(Mobject):
    def __init__(
        self,
        frame_shape: tuple[float, float] = FRAME_SHAPE,
        center_point: Vect3 = ORIGIN,
        # Field of view in the y direction
        fovy: float = 45 * DEG,
        euler_axes: str = "zxz",
        # This keeps it ordered first in a scene
        z_index=-1,
        **kwargs,
    ):
        super().__init__(z_index=z_index, **kwargs)

        self.uniforms["orientation"] = Rotation.identity().as_quat()
        self.uniforms["fovy"] = fovy

        self.default_orientation = Rotation.identity()
        self.view_matrix = np.identity(4)
        self.id4x4 = np.identity(4)
        self.camera_location = OUT  # This will be updated by set_points
        self.euler_axes = euler_axes

        self.set_points(np.array([ORIGIN, LEFT, RIGHT, DOWN, UP]))
        self.set_width(frame_shape[0], stretch=True)
        self.set_height(frame_shape[1], stretch=True)
        self.move_to(center_point)

    def set_orientation(self, rotation: Rotation):
        self.uniforms["orientation"][:] = rotation.as_quat()
        return self

    def get_orientation(self):
        return Rotation.from_quat(self.uniforms["orientation"])

    def make_orientation_default(self):
        self.default_orientation = self.get_orientation()
        return self

    def to_default_state(self):
        self.set_shape(*FRAME_SHAPE)
        self.center()
        self.set_orientation(self.default_orientation)
        return self

    def get_euler_angles(self) -> np.ndarray:
        orientation = self.get_orientation()
        if np.isclose(orientation.as_quat(), [0, 0, 0, 1]).all():
            return np.zeros(3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # Ignore UserWarnings
            angles = orientation.as_euler(self.euler_axes)[::-1]
        # Handle Gimble lock case
        if self.euler_axes == "zxz":
            if np.isclose(angles[1], 0, atol=1e-2):
                angles[0] = angles[0] + angles[2]
                angles[2] = 0
            if np.isclose(angles[1], PI, atol=1e-2):
                angles[0] = angles[0] - angles[2]
                angles[2] = 0
        return angles

    def get_theta(self):
        return self.get_euler_angles()[0]

    def get_phi(self):
        return self.get_euler_angles()[1]

    def get_gamma(self):
        return self.get_euler_angles()[2]

    def get_scale(self):
        return self.get_height() / FRAME_SHAPE[1]

    def get_inverse_camera_rotation_matrix(self):
        return self.get_orientation().as_matrix().T

    def get_view_matrix(self, refresh=False):
        """
        Returns a 4x4 for the affine transformation mapping a point
        into the camera's internal coordinate system
        """
        if self._data_has_changed:
            shift = self.id4x4.copy()
            rotation = self.id4x4.copy()

            scale = self.get_scale()
            shift[:3, 3] = -self.get_center()
            rotation[:3, :3] = self.get_inverse_camera_rotation_matrix()
            np.dot(rotation, shift, out=self.view_matrix)
            if scale > 0:
                self.view_matrix[:3, :4] /= scale

        return self.view_matrix

    def get_inv_view_matrix(self):
        return np.linalg.inv(self.get_view_matrix())

    @Mobject.affects_data
    def interpolate(self, mobject1, mobject2, alpha, *args, **kwargs):
        super().interpolate(mobject1, mobject2, alpha, *args, **kwargs)
        # q and -q are the same orientation; blend toward whichever is closer
        # to the start so the camera turns the short way around
        if "orientation" in mobject1.uniforms and "orientation" in mobject2.uniforms:
            q1 = mobject1.uniforms["orientation"]
            q2 = mobject2.uniforms["orientation"]
            if np.dot(q1, q2) < 0:
                self.uniforms["orientation"] = (1 - alpha) * q1 - alpha * q2

    @Mobject.affects_data
    def rotate(self, angle: float, axis: np.ndarray = OUT, **kwargs):
        rot = Rotation.from_rotvec(angle * normalize(axis))
        self.set_orientation(rot * self.get_orientation())
        return self

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
        units: float = RADIANS
    ):
        eulers = self.get_euler_angles()  # theta, phi, gamma
        for i, var in enumerate([theta, phi, gamma]):
            if var is not None:
                eulers[i] = var * units
        if all(eulers == 0):
            rot = Rotation.identity()
        else:
            rot = Rotation.from_euler(self.euler_axes, eulers[::-1])
        self.set_orientation(rot)
        return self

    def increment_euler_angles(
        self,
        dtheta: float = 0,
        dphi: float = 0,
        dgamma: float = 0,
        units: float = RADIANS
    ):
        angles = self.get_euler_angles()
        new_angles = angles + np.array([dtheta, dphi, dgamma]) * units

        # Limit range for phi
        if self.euler_axes == "zxz":
            new_angles[1] = clip(new_angles[1], 0, PI)
        elif self.euler_axes == "zxy":
            new_angles[1] = clip(new_angles[1], -PI / 2, PI / 2)

        new_rot = Rotation.from_euler(self.euler_axes, new_angles[::-1])
        self.set_orientation(new_rot)
        return self

    def increment_mirrored_euler_angles(
        self,
        dtheta: float = 0,
        dphi: float = 0,
        units: float = RADIANS
    ):
        """
        Mirror of increment_euler_angles for tilting below the default view:
        phi is allowed to go negative, down to -PI, but cannot be raised above
        max(current phi, 0), so the view never jumps when this takes over.
        """
        if self.euler_axes != "zxz":
            return self.increment_euler_angles(dtheta=dtheta, dphi=dphi, units=units)

        theta, phi, gamma = self.get_euler_angles()
        # as_euler always reports phi in [0, PI], so a negative phi comes back
        # as the equivalent (theta + PI, -phi, gamma + PI).  Undo that whenever
        # it leaves gamma closer to zero.
        if abs((gamma + PI) % TAU - PI) > PI / 2:
            theta, phi, gamma = theta - PI, -phi, gamma - PI
        # Looking straight up, phi = PI and -PI are the same orientation;
        # take the mirrored one so this stays at its lower limit.
        if np.isclose(phi, PI, atol=1e-2):
            phi = -PI

        theta += dtheta * units
        phi = clip(phi + dphi * units, -PI, max(phi, 0))

        new_rot = Rotation.from_euler(self.euler_axes, [gamma, phi, theta])
        self.set_orientation(new_rot)
        return self

    def set_euler_axes(self, seq: str):
        self.euler_axes = seq

    def reorient(
        self,
        theta_degrees: float | None = None,
        phi_degrees: float | None = None,
        gamma_degrees: float | None = None,
        center: Vect3 | tuple[float, float, float] | None = None,
        height: float | None = None
    ):
        """
        Shortcut for set_euler_angles, defaulting to taking
        in angles in degrees
        """
        self.set_euler_angles(theta_degrees, phi_degrees, gamma_degrees, units=DEG)
        if center is not None:
            self.move_to(np.array(center))
        if height is not None:
            self.set_height(height)
        return self

    def set_theta(self, theta: float):
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float):
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float):
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta: float, units=RADIANS):
        self.increment_euler_angles(dtheta=dtheta, units=units)
        return self

    def increment_phi(self, dphi: float, units=RADIANS):
        self.increment_euler_angles(dphi=dphi, units=units)
        return self

    def increment_gamma(self, dgamma: float, units=RADIANS):
        self.increment_euler_angles(dgamma=dgamma, units=units)
        return self

    def add_ambient_rotation(self, angular_speed=1 * DEG):
        self.add_updater(lambda m, dt: m.increment_theta(angular_speed * dt))
        return self

    @Mobject.affects_data
    def set_focal_distance(self, focal_distance: float):
        self.uniforms["fovy"] = 2 * math.atan(0.5 * self.get_height() / focal_distance)
        return self

    @Mobject.affects_data
    def set_field_of_view(self, field_of_view: float):
        self.uniforms["fovy"] = field_of_view
        return self

    def get_shape(self):
        return (self.get_width(), self.get_height())

    def get_aspect_ratio(self):
        width, height = self.get_shape()
        return width / height

    def get_center(self) -> np.ndarray:
        # Assumes first point is at the center
        return self.get_points()[0]

    def get_width(self) -> float:
        points = self.get_points()
        return points[2, 0] - points[1, 0]

    def get_height(self) -> float:
        points = self.get_points()
        return points[4, 1] - points[3, 1]

    def get_focal_distance(self) -> float:
        return 0.5 * self.get_height() / math.tan(0.5 * self.uniforms["fovy"])

    def get_field_of_view(self) -> float:
        return self.uniforms["fovy"]

    def get_implied_camera_location(self) -> np.ndarray:
        if self._data_has_changed:
            to_camera = self.get_inverse_camera_rotation_matrix()[2]
            dist = self.get_focal_distance()
            self.camera_location = self.get_center() + dist * to_camera
        return self.camera_location

    def to_fixed_frame_point(self, point: Vect3, relative: bool = False):
        view = self.get_view_matrix()
        point4d = [*point, 0 if relative else 1]
        return np.dot(point4d, view.T)[:3]

    def from_fixed_frame_point(self, point: Vect3, relative: bool = False):
        inv_view = self.get_inv_view_matrix()
        point4d = [*point, 0 if relative else 1]
        return np.dot(point4d, inv_view.T)[:3]
