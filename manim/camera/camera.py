from __future__ import annotations

__all__ = ["Camera"]

import copy
import itertools as it
import operator as op
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist
from collections.abc import Iterable
from functools import reduce
from typing import Any, Callable
import base64
from io import BytesIO

from js import document, Uint8ClampedArray, Image as JSImage

from .. import config, logger
from ..constants import *
from ..mobject.mobject import Mobject
from ..mobject.types.image_mobject import AbstractImageMobject
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.color import ManimColor, ParsableManimColor, color_to_int_rgba
from ..utils.family import extract_mobject_family_members
from ..utils.images import get_full_raster_image_path
from ..utils.iterables import list_difference_update
from ..utils.space_ops import angle_of_vector


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

        self.canvas = document.createElement("canvas")
        self.canvas.width = self.pixel_width
        self.canvas.height = self.pixel_height
        self.ctx = self.canvas.getContext("2d")

        self.init_background()
        self.resize_frame_shape()
    
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
    
    def img_to_base64(self, image: Image.Image) -> str:
        """Convierte una imagen PIL a una cadena base64.

        Parameters
        ----------
        image : Image.Image
            Imagen PIL a convertir.

        Returns
        -------
        str
            Cadena base64 de la imagen.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        data = buffered.read()
        b64_data = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"
    
    async def get_image(self, base64_string: str):
        """Convierte un arreglo de píxeles (np.ndarray) a un objeto Image del canvas.

        Parameters
        ----------
        pixel_array : np.ndarray
            Arreglo de forma (H, W, 4) con datos RGBA.

        Returns
        -------
        ImageData
            Objeto ImageData listo para usarse en ctx.drawImage.
        """
        if base64_string in self.cached_b64:
            return self.cached_b64[base64_string]
        js_image = JSImage.new()
        js_image.src = base64_string
        await js_image.decode()
        self.cached_b64[base64_string] = js_image
        return js_image

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

    async def reset_pixel_array(self, new_height: float, new_width: float):
        self.pixel_width = new_width
        self.pixel_height = new_height
        self.init_background()
        self.resize_frame_shape()
        await self.reset()
    
    @property
    def pixel_array(self) -> np.ndarray:
        """Arreglo de píxeles del canvas."""
        data = self.ctx.getImageData(
            0,
            0,
            self.canvas.width,
            self.canvas.height,
        ).data
        return np.array(data, dtype=self.pixel_array_dtype).reshape(
            (self.pixel_height, self.pixel_width, self.n_channels)
        )

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

    async def reset(self):
        await self.set_pixel_array(self.background)
        return self

    async def set_frame_to_background(self, background):
        """Establece el arreglo de píxeles actual al fondo proporcionado."""
        await self.set_pixel_array(background)

    async def set_pixel_array(self, pixel_array: np.ndarray | list | tuple):
        img = Image.fromarray(
            np.array(pixel_array, dtype=self.pixel_array_dtype),
            mode=self.image_mode,
        )
        b64_string = self.img_to_base64(img)
        js_image = await self.get_image(b64_string)
        self.undo_canvas_transform()
        self.ctx.clearRect(0, 0, self.canvas.width, self.canvas.height)
        self.ctx.drawImage(js_image, 0, 0, self.canvas.width, self.canvas.height)

    async def capture_mobject(self, mobject: Mobject, **kwargs: Any):
        return await self.capture_mobjects([mobject], **kwargs)

    async def capture_mobjects(self, mobjects: Iterable[Mobject], **kwargs):
        mobjects = self.get_mobjects_to_display(mobjects, **kwargs)
        for group_type, group in it.groupby(mobjects, self.type_or_raise):
            await self.display_funcs[group_type](list(group))

    def type_or_raise(self, mobject: Mobject):
        async def fallback(group):
            pass
        self.display_funcs = {
            VMobject: self.display_multiple_non_background_colored_vmobjects,
            AbstractImageMobject: self.display_multiple_image_mobjects,
            Mobject: fallback
        }
        for _type in self.display_funcs:
            if isinstance(mobject, _type):
                return _type
        raise TypeError(f"Displaying an object of class {type(mobject)} is not supported")

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

    async def display_multiple_non_background_colored_vmobjects(self, vmobjects: list):
        ctx = self.ctx
        self.apply_canvas_transform()

        for vmobject in vmobjects:
            self.set_canvas_path(ctx, vmobject)
            self.apply_stroke(ctx, vmobject, background=True)
            self.apply_fill(ctx, vmobject)
            self.apply_stroke(ctx, vmobject)

    def set_canvas_path(self, ctx, vmobject: VMobject):
        points = self.transform_points_pre_display(vmobject, vmobject.points)
        if len(points) == 0:
            return

        ctx.beginPath()
        subpaths = vmobject.gen_subpaths_from_points_2d(points)
        for subpath in subpaths:
            if len(subpath) == 0:
                continue
            start = subpath[0]
            ctx.moveTo(*start[:2])
            quads = vmobject.gen_cubic_bezier_tuples_from_points(subpath)
            for _, p1, p2, p3 in quads:
                ctx.bezierCurveTo(*p1[:2], *p2[:2], *p3[:2])
            if vmobject.consider_points_equals_2d(subpath[0], subpath[-1]):
                ctx.closePath()
    
    def set_canvas_fill_gradient(self, ctx, vmobject: VMobject, r: int, g: int, b: int, a: float):
        """Configura un gradiente de relleno en el contexto del canvas basado en los puntos del VMobject."""
        rgbas = self.get_fill_rgbas(vmobject)
        if len(rgbas) == 1:
            ctx.fillStyle = f"rgba({r}, {g}, {b}, {a})"
            return

        points = vmobject.get_gradient_start_and_end_points()
        points = self.transform_points_pre_display(vmobject, points)
        x0, y0 = points[0][:2]
        x1, y1 = points[1][:2]
        gradient = ctx.createLinearGradient(x0, y0, x1, y1)

        step = 1.0 / (len(rgbas) - 1)
        for i, rgba in enumerate(rgbas):
            r, g, b, a = [int(c * 255) if j < 3 else c for j, c in enumerate(rgba)]
            gradient.addColorStop(i * step, f"rgba({r}, {g}, {b}, {a})")

        ctx.fillStyle = gradient

    def apply_fill(self, ctx, vmobject: VMobject):
        rgba = vmobject.get_fill_rgbas()[0]
        r, g, b, a = [int(c * 255) if i < 3 else c for i, c in enumerate(rgba)]
        self.set_canvas_fill_gradient(ctx, vmobject, r, g, b, a)
        ctx.fill()
    
    def apply_canvas_transform(self):
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

        self.ctx.setTransform(scale_x, 0, 0, scale_y, translate_x, translate_y)

    def get_fill_rgbas(self, vmobject: VMobject):
        return vmobject.get_fill_rgbas()
    
    def get_stroke_rgbas(self, vmobject: VMobject, background: bool = False):
        return vmobject.get_stroke_rgbas(background=background)
    
    def undo_canvas_transform(self):
        """Reestablece la matriz de transformación del contexto de canvas a la identidad."""
        self.ctx.setTransform(1, 0, 0, 1, 0, 0)
    
    def set_canvas_stroke_gradient(self, ctx, vmobject: VMobject, background: bool = False):
        """Configura un gradiente lineal de trazo en el contexto del canvas basado en los puntos del VMobject."""
        rgbas = self.get_stroke_rgbas(vmobject, background=background)
        if len(rgbas) == 1:
            r, g, b, a = [int(c * 255) if i < 3 else c for i, c in enumerate(rgbas[0])]
            ctx.strokeStyle = f"rgba({r}, {g}, {b}, {a})"
            return

        points = vmobject.get_gradient_start_and_end_points()
        points = self.transform_points_pre_display(vmobject, points)
        x0, y0 = points[0][:2]
        x1, y1 = points[1][:2]
        gradient = ctx.createLinearGradient(x0, y0, x1, y1)

        step = 1.0 / (len(rgbas) - 1)
        for i, rgba in enumerate(rgbas):
            r, g, b, a = [int(c * 255) if j < 3 else c for j, c in enumerate(rgba)]
            gradient.addColorStop(i * step, f"rgba({r}, {g}, {b}, {a})")

        ctx.strokeStyle = gradient

    def apply_stroke(self, ctx, vmobject: VMobject, background=False):
        width = vmobject.get_stroke_width(background)
        if width == 0:
            return
        self.set_canvas_stroke_gradient(ctx, vmobject, background=background)
        ctx.lineWidth = width / 100
        ctx.stroke()

    async def display_multiple_image_mobjects(self, image_mobjects: list):
        for image_mobject in image_mobjects:
            await self.display_image_mobject(image_mobject)

    async def display_image_mobject(self, image_mobject: AbstractImageMobject):
        self.undo_canvas_transform()
        corner_coords = self.points_to_pixel_coords(image_mobject, image_mobject.points)
        ul_coords, ur_coords, dl_coords, _ = corner_coords
        right_vect = ur_coords - ul_coords
        down_vect = dl_coords - ul_coords
        center_coords = ul_coords + (right_vect + down_vect) / 2

        sub_image = Image.fromarray(image_mobject.get_pixel_array(), mode="RGBA")
        pixel_width = max(int(pdist([ul_coords, ur_coords]).item()), 1)
        pixel_height = max(int(pdist([ul_coords, dl_coords]).item()), 1)
        sub_image = sub_image.resize((pixel_width, pixel_height), resample=image_mobject.resampling_algorithm)

        angle = angle_of_vector(right_vect)
        adjusted_angle = -int(360 * angle / TAU)
        if adjusted_angle != 0:
            sub_image = sub_image.rotate(adjusted_angle, resample=image_mobject.resampling_algorithm, expand=1)

        full_image = Image.fromarray(np.zeros((self.pixel_height, self.pixel_width)), mode="RGBA")
        new_ul_coords = center_coords - np.array(sub_image.size) / 2
        new_ul_coords = new_ul_coords.astype(int)
        full_image.paste(sub_image, box=(
            new_ul_coords[0],
            new_ul_coords[1],
            new_ul_coords[0] + sub_image.size[0],
            new_ul_coords[1] + sub_image.size[1],
        ))
        b64_string = self.img_to_base64(full_image)
        js_image = await self.get_image(b64_string)
        self.ctx.drawImage(
            js_image,
            0, 0,
            self.pixel_width, self.pixel_height,
        )

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
