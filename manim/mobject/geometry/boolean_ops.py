"""Boolean operations for two-dimensional mobjects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from js import window

from manim import config
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject

if TYPE_CHECKING:
    from typing import Any

    from manim.typing import Point2DLike_Array, Point3D_Array, Point3DLike_Array

__all__ = ["Union", "Intersection", "Difference", "Exclusion"]


class _BooleanOps(VMobject, metaclass=ConvertToOpenGL):
    """This class contains some helper functions which
    helps to convert to and from skia objects and manim
    objects (:class:`~.VMobject`).
    """

    def _convert_2d_to_3d_array(
        self,
        points: Point2DLike_Array | Point3DLike_Array,
        z_dim: float = 0.0,
    ) -> Point3D_Array:
        """Converts an iterable with coordinates in 2D to 3D by adding
        :attr:`z_dim` as the Z coordinate.

        Parameters
        ----------
        points
            An iterable of points.
        z_dim
            Default value for the Z coordinate.

        Returns
        -------
        Point3D_Array
            A list of the points converted to 3D.

        Example
        -------
        >>> a = _BooleanOps()
        >>> p = [(1, 2), (3, 4)]
        >>> a._convert_2d_to_3d_array(p)
        array([[1., 2., 0.],
               [3., 4., 0.]])
        """
        list_of_points = list(points)
        for i, point in enumerate(list_of_points):
            if len(point) == 2:
                list_of_points[i] = np.array(list(point) + [z_dim])
        return np.asarray(list_of_points)

    def _convert_vmobject_to_skia_path(self, vmobject: VMobject) -> Any:
        """Converts a :class:`~.VMobject` to SkiaPath. This method only works for
        cairo renderer because it treats the points as Cubic beizer curves.

        Parameters
        ----------
        vmobject:
            The :class:`~.VMobject` to convert from.

        Returns
        -------
        SkiaPath
            The converted path.
        """
        PathKit = window.PathKit

        path = PathKit.NewPath()

        if not np.all(np.isfinite(vmobject.points)):
            points = np.zeros((1, 3))  # point invalid?
        else:
            points = vmobject.points

        if len(points) == 0:  # what? No points so return empty path
            return path

        subpaths = vmobject.gen_subpaths_from_points_2d(points)  # type: ignore[assignment]
        for subpath in subpaths:
            quads = vmobject.gen_cubic_bezier_tuples_from_points(subpath)
            start = subpath[0]
            path.moveTo(*start[:2])
            for _p0, p1, p2, p3 in quads:
                path.cubicTo(*p1[:2], *p2[:2], *p3[:2])

            if vmobject.consider_points_equals_2d(subpath[0], subpath[-1]):
                path.close()

        return path

    def _convert_skia_path_to_vmobject(self, path: Any) -> VMobject:
        """Converts SkiaPath back to VMobject.
        Parameters
        ----------
        path:
            The SkiaPath to convert.

        Returns
        -------
        VMobject:
            The converted VMobject.
        """
        vmobject = self
        current_path_start = np.array([0, 0, 0])
        PathKit = window.PathKit

        for cmd in path.toCmds():
            path_verb = cmd[0]
            points = np.array(cmd[1:]).reshape(-1, 2)
            if path_verb == PathKit.MOVE_VERB:
                parts = self._convert_2d_to_3d_array(points)
                for part in parts:
                    current_path_start = part
                    vmobject.start_new_path(part)
                    # vmobject.move_to(*part)
            elif path_verb == PathKit.CUBIC_VERB:
                n1, n2, n3 = self._convert_2d_to_3d_array(points)
                vmobject.add_cubic_bezier_curve_to(n1, n2, n3)
            elif path_verb == PathKit.LINE_VERB:
                parts = self._convert_2d_to_3d_array(points)
                vmobject.add_line_to(parts[0])
            elif path_verb == PathKit.CLOSE_VERB:
                vmobject.add_line_to(current_path_start)
            elif path_verb == PathKit.QUAD_VERB:
                n1, n2 = self._convert_2d_to_3d_array(points)
                vmobject.add_quadratic_bezier_curve_to(n1, n2)
            else:
                raise Exception(f"Unsupported: {path_verb}")
        return vmobject


class Union(_BooleanOps):
    """Union of two or more :class:`~.VMobject` s. This returns the common region of
    the :class:`~VMobject` s.

    Parameters
    ----------
    vmobjects
        The :class:`~.VMobject` s to find the union of.

    Raises
    ------
    ValueError
        If less than 2 :class:`~.VMobject` s are passed.

    Example
    -------
    .. manim:: UnionExample
        :save_last_frame:

        class UnionExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Union(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0.3, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, *vmobjects: VMobject, **kwargs: Any) -> None:
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Union.")
        super().__init__(**kwargs)
        paths = []
        for vmobject in vmobjects:
            paths.append(self._convert_vmobject_to_skia_path(vmobject))
        PathKit = window.PathKit
        builder = PathKit.SkOpBuilder.new()
        op = PathKit.PathOp.UNION
        for path in paths:
            builder.add(path, op)
        outpen = builder.resolve()
        self._convert_skia_path_to_vmobject(outpen)
        # Free memory
        outpen.delete()
        builder.delete()
        for path in paths:
            path.delete()


class Difference(_BooleanOps):
    """Subtracts one :class:`~.VMobject` from another one.

    Parameters
    ----------
    subject
        The 1st :class:`~.VMobject`.
    clip
        The 2nd :class:`~.VMobject`

    Example
    -------
    .. manim:: DifferenceExample
        :save_last_frame:

        class DifferenceExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Difference(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, subject: VMobject, clip: VMobject, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        PathKit = window.PathKit
        op = PathKit.PathOp.DIFFERENCE
        path1 = self._convert_vmobject_to_skia_path(subject)
        path2 = self._convert_vmobject_to_skia_path(clip)
        path1.op(path2, op)
        self._convert_skia_path_to_vmobject(path1)
        # Free memory
        path1.delete()
        path2.delete()


class Intersection(_BooleanOps):
    """Find the intersection of two :class:`~.VMobject` s.
    This keeps the parts covered by both :class:`~.VMobject` s.

    Parameters
    ----------
    vmobjects
        The :class:`~.VMobject` to find the intersection.

    Raises
    ------
    ValueError
        If less the 2 :class:`~.VMobject` are passed.

    Example
    -------
    .. manim:: IntersectionExample
        :save_last_frame:

        class IntersectionExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Intersection(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, *vmobjects: VMobject, **kwargs: Any) -> None:
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Intersection.")

        super().__init__(**kwargs)
        paths = []
        for vmobject in vmobjects:
            paths.append(self._convert_vmobject_to_skia_path(vmobject))
        PathKit = window.PathKit
        builder = PathKit.SkOpBuilder.new()
        op = PathKit.PathOp.INTERSECT
        for path in paths:
            builder.add(path, op)
        outpen = builder.resolve()
        self._convert_skia_path_to_vmobject(outpen)
        # Free memory
        outpen.delete()
        builder.delete()
        for path in paths:
            path.delete()


class Exclusion(_BooleanOps):
    """Find the XOR between two :class:`~.VMobject`.
    This creates a new :class:`~.VMobject` consisting of the region
    covered by exactly one of them.

    Parameters
    ----------
    subject
        The 1st :class:`~.VMobject`.
    clip
        The 2nd :class:`~.VMobject`

    Example
    -------
    .. manim:: IntersectionExample
        :save_last_frame:

        class IntersectionExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Exclusion(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0.4, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, subject: VMobject, clip: VMobject, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        PathKit = window.PathKit
        op = PathKit.PathOp.XOR
        path1 = self._convert_vmobject_to_skia_path(subject)
        path2 = self._convert_vmobject_to_skia_path(clip)
        path1.op(path2, op)
        self._convert_skia_path_to_vmobject(path1)
        # Free memory
        path1.delete()
        path2.delete()
