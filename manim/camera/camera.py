from __future__ import annotations

__all__ = ["Camera"]

import numpy as np
from PIL import Image
from collections.abc import Iterable
from typing import Any

from js import Uint8ClampedArray, ImageData, Uint8ClampedArray, Object, CustomEvent
from pyodide.ffi import to_js

from .. import config
from ..constants import *
from ..mobject.mobject import Mobject
from ..mobject.types.image_mobject import AbstractImageMobject
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.color import ManimColor, ParsableManimColor, color_to_int_rgba
from ..utils.family import extract_mobject_family_members
from ..utils.images import get_full_raster_image_path
from ..utils.iterables import list_difference_update


class VMobjectData:
    def __init__(
        self,
        vmobject: VMobject,
        camera: Camera,
    ) -> None:
        self.vmobject = vmobject
        self.camera = camera
        self.points = self.camera.transform_points_pre_display(vmobject, vmobject.points)
        self.points = [[[p[:2].tolist() for p in tuple] for tuple in self.vmobject.gen_cubic_bezier_tuples_from_points(subpath)] for subpath in self.vmobject.gen_subpaths_from_points_2d(self.points)]
        self.transform = self.camera.get_canvas_transform()
        self.stroke_rgbas = self.camera.get_stroke_rgbas(vmobject).tolist()
        self.background_stroke_rgbas = self.camera.get_stroke_rgbas(vmobject, background=True).tolist()
        self.fill_rgbas = self.camera.get_fill_rgbas(vmobject).tolist()
        self.gradient_start_and_end_points = vmobject.get_gradient_start_and_end_points()
        self.gradient_start_and_end_points = np.array(self.camera.transform_points_pre_display(
            vmobject,
            self.gradient_start_and_end_points,
        )).tolist()
        self.stroke_width = vmobject.get_stroke_width()
        self.background_stroke_width = vmobject.get_stroke_width(background=True)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "VMobjectData",
            "points": self.points,
            "transform": self.transform,
            "stroke_rgbas": self.stroke_rgbas,
            "background_stroke_rgbas": self.background_stroke_rgbas,
            "fill_rgbas": self.fill_rgbas,
            "gradient_start_and_end_points": self.gradient_start_and_end_points,
            "stroke_width": self.stroke_width,
            "background_stroke_width": self.background_stroke_width,
        }


class AbstractImageData:
    def __init__(
        self,
        image_mobject: AbstractImageMobject,
        camera: Camera,
    ) -> None:
        self.image_mobject = image_mobject
        self.camera = camera
        pixel_array = image_mobject.get_pixel_array()
        p0, p1, p2, p3 = self.camera.points_to_pixel_coords(
            image_mobject,
            image_mobject.points,
        )
        dx1 = p1[0] - p0[0]
        dy1 = p1[1] - p0[1]
        dx2 = p2[0] - p0[0]
        dy2 = p2[1] - p0[1]
        w = pixel_array.shape[1]
        h = pixel_array.shape[0]
        a = dx1 / w
        b = dy1 / w
        c = dx2 / h
        d = dy2 / h
        e = p0[0]
        f = p0[1]
        self.transform = [a, b, c, d, e, f]
        self.image_data = camera.get_image_data(np.array(pixel_array, dtype=camera.pixel_array_dtype))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "AbstractImageData",
            "transform": self.transform,
            "image_data": self.image_data,
        }


class Camera:
    def __init__(
        self,
        background_image: str | None = None,
        frame_center: np.ndarray = ORIGIN,
        image_mode: str = "RGBA",
        n_channels: int = 4,
        pixel_array_dtype: str = "uint8",
        use_z_index: bool = True,
        background: np.ndarray | None = None,
        pixel_height: int | None = None,
        pixel_width: int | None = None,
        frame_height: float | None = None,
        frame_width: float | None = None,
        frame_rate: float | None = None,
        background_color: ParsableManimColor | None = None,
        background_opacity: float | None = None,
        **kwargs,
    ):
        self.element = None
        self.event_name = None
        self.cached_b64 = {}
        self.background_image = background_image
        self.frame_center = frame_center
        self.image_mode = image_mode
        self.n_channels = n_channels
        self.pixel_array_dtype = pixel_array_dtype
        self.use_z_index = use_z_index
        self.background = background

        if pixel_height is None:
            pixel_height = config["pixel_height"]
        self.pixel_height = pixel_height

        if pixel_width is None:
            pixel_width = config["pixel_width"]
        self.pixel_width = pixel_width

        if frame_height is None:
            frame_height = config["frame_height"]
        self.frame_height = frame_height

        if frame_width is None:
            frame_width = config["frame_width"]
        self.frame_width = frame_width

        if frame_rate is None:
            frame_rate = config["frame_rate"]
        self.frame_rate = frame_rate

        if background_color is None:
            self._background_color = ManimColor.parse(config["background_color"])
        else:
            self._background_color = ManimColor.parse(background_color)
        if background_opacity is None:
            self._background_opacity = config["background_opacity"]
        else:
            self._background_opacity = background_opacity

        self.max_allowable_norm = config["frame_width"]
        self.rgb_max_val = np.iinfo(self.pixel_array_dtype).max

        self.init_background()
        self.resize_frame_shape()
    
    def init_element(self, element, event_name):
        self.element = element
        self.event_name = event_name
    
    @property
    def background_color(self) -> ManimColor:
        """Color de fondo del canvas."""
        return self._background_color
    
    @background_color.setter
    def background_color(self, value: ParsableManimColor):
        """Establece el color de fondo del canvas.

        Parameters
        ----------
        value : ParsableManimColor
            Color de fondo a establecer.
        """
        self._background_color = ManimColor.parse(value)
        self.init_background()
    
    @property
    def background_opacity(self) -> float:
        """Opacidad del fondo del canvas."""
        return self._background_opacity
    
    @background_opacity.setter
    def background_opacity(self, value: float):
        """Establece la opacidad del fondo del canvas.

        Parameters
        ----------
        value : float
            Opacidad del fondo a establecer.
        """
        if not (0 <= value <= 1):
            raise ValueError("La opacidad debe estar entre 0 y 1.")
        self._background_opacity = value
        self.init_background()
    
    def get_image_data(self, arr: np.ndarray):
        """Convierte un arreglo de píxeles (np.ndarray) a un objeto ImageData del canvas.

        Parameters
        ----------
        pixel_array : np.ndarray
            Arreglo de forma (H, W, 4) con datos RGBA.

        Returns
        -------
        ImageData
            Objeto ImageData listo para usarse en ctx.putImageData().
        """
        js_buf = Uint8ClampedArray.new(arr.tobytes())
        img = ImageData.new(js_buf, arr.shape[1], arr.shape[0])
        return img

    def init_background(self):
        height = self.pixel_height
        width = self.pixel_width
        if self.background_image is not None:
            path = get_full_raster_image_path(self.background_image)
            image = Image.open(path).convert(self.image_mode)
            self.background = np.array(image)[:height, :width]
            self.background = self.background.astype(self.pixel_array_dtype)
        else:
            background_rgba = color_to_int_rgba(
                self.background_color,
                self.background_opacity,
            )
            self.background = np.zeros(
                (height, width, self.n_channels),
                dtype=self.pixel_array_dtype,
            )
            self.background[:, :] = background_rgba

    def resize_frame_shape(self, fixed_dimension: int = 0):
        pixel_height = self.pixel_height
        pixel_width = self.pixel_width
        frame_height = self.frame_height
        frame_width = self.frame_width
        aspect_ratio = pixel_width / pixel_height
        if fixed_dimension == 0:
            frame_height = frame_width / aspect_ratio
        else:
            frame_width = aspect_ratio * frame_height
        self.frame_height = frame_height
        self.frame_width = frame_width

    def capture_mobject(self, mobject: Mobject, **kwargs: Any):
        return self.capture_mobjects([mobject], **kwargs)

    def capture_mobjects(self, mobjects: Iterable[Mobject], **kwargs):
        if self.element is None or self.event_name is None:
            return
        mobjects = self.get_mobjects_to_display(mobjects, **kwargs)
        data = [Object.fromEntries(to_js({
            "type": "BackgroundData",
            "background_color": self.background_color.to_rgba().tolist(),
            "background_opacity": self.background_opacity,
        }))]
        for mobject in mobjects:
            if isinstance(mobject, VMobject):
                data.append(Object.fromEntries(to_js(VMobjectData(mobject, self).to_dict())))
            elif isinstance(mobject, AbstractImageMobject):
                data.append(Object.fromEntries(to_js(AbstractImageData(mobject, self).to_dict())))
            else:
                pass
        event = CustomEvent.new(self.event_name, Object.fromEntries(to_js({
            "detail": data,
        })))
        self.element.dispatchEvent(event)

    def get_mobjects_to_display(self, mobjects: Iterable[Mobject], include_submobjects: bool = True, excluded_mobjects: list | None = None):
        if include_submobjects:
            mobjects = extract_mobject_family_members(
                mobjects,
                use_z_index=self.use_z_index,
                only_those_with_points=True,
            )
            if excluded_mobjects:
                all_excluded = extract_mobject_family_members(
                    excluded_mobjects,
                    use_z_index=self.use_z_index,
                )
                mobjects = list_difference_update(mobjects, all_excluded)
        return list(mobjects)
    
    def get_stroke_rgbas(self, mobject: VMobject, background: bool = False):
        return mobject.get_stroke_rgbas(background)
    
    def get_fill_rgbas(self, mobject: VMobject):
        return mobject.get_fill_rgbas()
    
    def get_canvas_transform(self):
        """Configura la matriz de transformación en el contexto de canvas para igualar la transformación que hacía Cairo."""
        pw = self.pixel_width
        ph = self.pixel_height
        fw = self.frame_width
        fh = self.frame_height
        fc = self.frame_center

        scale_x = pw / fw
        scale_y = -ph / fh  # invertir Y
        translate_x = pw / 2 - fc[0] * scale_x
        translate_y = ph / 2 - fc[1] * scale_y

        return [scale_x, 0, 0, scale_y, translate_x, translate_y]

    def overlay_PIL_image(self, pixel_array: np.ndarray, image: Image):
        pixel_array[:, :] = np.array(
            Image.alpha_composite(Image.fromarray(pixel_array, mode=self.image_mode), image),
            dtype="uint8",
        )

    def transform_points_pre_display(self, mobject, points):
        if not np.all(np.isfinite(points)):
            points = np.zeros((1, 3))
        return points

    def points_to_pixel_coords(self, mobject, points):
        points = self.transform_points_pre_display(mobject, points)
        shifted_points = points - self.frame_center
        result = np.zeros((len(points), 2))
        width_mult = self.pixel_width / self.frame_width
        width_add = self.pixel_width / 2
        height_mult = -self.pixel_height / self.frame_height
        height_add = self.pixel_height / 2
        result[:, 0] = shifted_points[:, 0] * width_mult + width_add
        result[:, 1] = shifted_points[:, 1] * height_mult + height_add
        return result.astype("int")
