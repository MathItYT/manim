from __future__ import annotations

import typing

import numpy as np

from manim.utils.hashing import get_hash_from_play_call

from .. import config, logger
from ..camera.camera import Camera
from ..mobject.mobject import Mobject, _AnimationBuilder
from ..utils.iterables import list_update

if typing.TYPE_CHECKING:
    from typing import Any

    from manim.animation.animation import Animation
    from manim.scene.scene import Scene

    from ..typing import PixelArray

__all__ = ["CanvasRenderer"]


class CanvasRenderer:
    """A renderer using Cairo.

    num_plays : Number of play() functions in the scene.
    time: time elapsed since initialisation of scene.
    """

    def __init__(
        self,
        camera_class=None,
        skip_animations=False,
        **kwargs,
    ):
        # All of the following are set to EITHER the value passed via kwargs,
        # OR the value stored in the global config dict at the time of
        # _instance construction_.
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera_cls()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes = []
        self.num_plays = 0
        self.time = 0
        self.static_image = None

    def init_scene(self, scene):
        pass

    @property
    def canvas(self) -> Any:
        """The canvas element of the renderer."""
        return self.camera.canvas
    
    @property
    def ctx(self) -> Any:
        """The context of the canvas element of the renderer."""
        return self.camera.ctx

    async def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs,
    ):
        # Reset skip_animations to the original state.
        # Needed when rendering only some animations, and skipping others.
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        scene.compile_animation_data(*args, **kwargs)

        if self.skip_animations:
            logger.debug(f"Skipping animation {self.num_plays}")
            hash_current_animation = None
            self.time += scene.duration
        else:
            if config["disable_caching"]:
                logger.info("Caching disabled.")
                hash_current_animation = f"uncached_{self.num_plays:05}"
            else:
                hash_current_animation = get_hash_from_play_call(
                    scene,
                    self.camera,
                    scene.animations,
                    scene.mobjects,
                )
        self.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )

        scene.begin_animations()

        # Save a static image, to avoid rendering non moving objects.
        self.save_static_frame_data(scene, scene.static_mobjects)

        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene, mobjects=scene.moving_mobjects)
            # self.duration stands for the total run time of all the animations.
            # In this case, as there is only a wait, it will be the length of the wait.
            self.freeze_current_frame(scene.duration)
        else:
            await scene.play_internal()

        self.num_plays += 1
    
    @property
    def static_image(self) -> PixelArray | None:
        return None
    
    @static_image.setter
    def static_image(self, value: PixelArray | None):
        pass

    def update_frame(  # TODO Description in Docstring
        self,
        scene,
        mobjects: typing.Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs,
    ):
        """Update the frame.

        Parameters
        ----------
        scene

        mobjects
            list of mobjects

        include_submobjects

        ignore_skipping

        **kwargs

        """
        if self.skip_animations and not ignore_skipping:
            return
        if not mobjects:
            mobjects = list_update(
                scene.mobjects,
                scene.foreground_mobjects,
            )

        kwargs["include_submobjects"] = include_submobjects
        self.camera.capture_mobjects(mobjects, **kwargs)

    def render(self, scene, time, moving_mobjects):
        self.update_frame(scene, moving_mobjects)

    def get_frame(self) -> PixelArray:
        """
        Gets the current frame as NumPy array.

        Returns
        -------
        np.array
            NumPy array of pixel values of each pixel in screen.
            The shape of the array is height x width x 3
        """
        return np.array(self.camera.pixel_array)

    def add_frame(self, frame: np.ndarray, num_frames: int = 1):
        """
        Adds a frame to the video_file_stream

        Parameters
        ----------
        frame
            The frame to add, as a pixel array.
        num_frames
            The number of times to add frame.
        """
        pass

    def freeze_current_frame(self, duration: float):
        """Adds a static frame to the movie for a given duration. The static frame is the current frame.

        Parameters
        ----------
        duration
            [description]
        """
        pass

    def show_frame(self):
        """
        Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.update_frame(ignore_skipping=True)

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: typing.Iterable[Mobject],
    ) -> typing.Iterable[Mobject] | None:
        """Compute and save the static frame, that will be reused at each frame
        to avoid unnecessarily computing static mobjects.

        Parameters
        ----------
        scene
            The scene played.
        static_mobjects
            Static mobjects of the scene. If None, self.static_image is set to None

        Returns
        -------
        typing.Iterable[Mobject]
            The static image computed.
        """
        return np.zeros((scene.camera.pixel_height, scene.camera.pixel_width, 3), dtype=np.uint8)

    def update_skipping_status(self):
        """
        This method is used internally to check if the current
        animation needs to be skipped or not. It also checks if
        the number of animations that were played correspond to
        the number of animations that need to be played, and
        raises an EndSceneEarlyException if they don't correspond.
        """
        pass

    def scene_finished(self, scene):
        pass
