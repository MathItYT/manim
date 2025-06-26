"""Basic canvas for animations."""

from __future__ import annotations

from manim.utils.parameter_parsing import flatten_iterable_parameters

__all__ = ["Scene"]

import copy
import inspect
import random
import asyncio
from pyodide.ffi import create_proxy
import types
from queue import Queue

from typing import TYPE_CHECKING

import numpy as np

from manim.mobject.mobject import Mobject

from .. import config, logger
from ..animation.animation import Animation, Wait, prepare_animation
from ..camera.camera import Camera
from ..constants import *
from ..renderer.canvas_renderer import CanvasRenderer
from ..utils.exceptions import EndSceneEarlyException, RerunSceneException
from ..utils.family import extract_mobject_family_members
from ..utils.iterables import list_difference_update, list_update

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable

    from manim.mobject.mobject import _AnimationBuilder


class Scene:
    """A Scene is the canvas of your animation.

    The primary role of :class:`Scene` is to provide the user with tools to manage
    mobjects and animations.  Generally speaking, a manim script consists of a class
    that derives from :class:`Scene` whose :meth:`Scene.construct` method is overridden
    by the user's code.

    Mobjects are displayed on screen by calling :meth:`Scene.add` and removed from
    screen by calling :meth:`Scene.remove`.  All mobjects currently on screen are kept
    in :attr:`Scene.mobjects`.  Animations are played by calling :meth:`Scene.play`.

    A :class:`Scene` is rendered internally by calling :meth:`Scene.render`.  This in
    turn calls :meth:`Scene.setup`, :meth:`Scene.construct`, and
    :meth:`Scene.tear_down`, in that order.

    It is not recommended to override the ``__init__`` method in user Scenes.  For code
    that should be ran before a Scene is rendered, use :meth:`Scene.setup` instead.

    Examples
    --------
    Override the :meth:`Scene.construct` method with your code.

    .. code-block:: python

        class MyScene(Scene):
            def construct(self):
                self.play(Write(Text("Hello World!")))

    """

    def __init__(
        self,
        renderer: CanvasRenderer | None = None,
        camera_class: type[Camera] = Camera,
        always_update_mobjects: bool = False,
        random_seed: int | None = None,
        skip_animations: bool = False,
    ) -> None:
        self.camera_class = camera_class
        self.always_update_mobjects = always_update_mobjects
        self.random_seed = random_seed
        self.skip_animations = skip_animations

        self.animations = None
        self.stop_condition = None
        self.moving_mobjects = []
        self.static_mobjects = []
        self.duration = None
        self.last_t = None
        self.queue = Queue()
        self.skip_animation_preview = False
        self.meshes = []
        self.camera_target = ORIGIN
        self.widgets = []
        self.updaters = []
        self.point_lights = []
        self.ambient_light = None
        self.key_to_function_map = {}
        self.mouse_press_callbacks = []
        self.interactive_mode = False

        if renderer is None:
            self.renderer = CanvasRenderer(
                camera_class=self.camera_class,
                skip_animations=self.skip_animations,
            )
        else:
            self.renderer = renderer
        self.renderer.init_scene(self)

        self.mobjects = []
        # TODO, remove need for foreground mobjects
        self.foreground_mobjects = []
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    @property
    def camera(self):
        return self.renderer.camera

    @property
    def time(self) -> float:
        """The time since the start of the scene."""
        return self.renderer.time

    def __deepcopy__(self, clone_from_id):
        cls = self.__class__
        result = cls.__new__(cls)
        clone_from_id[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["renderer", "time_progression"]:
                continue
            if k == "camera_class":
                setattr(result, k, v)
            setattr(result, k, copy.deepcopy(v, clone_from_id))
        result.mobject_updater_lists = []

        # Update updaters
        for mobject in self.mobjects:
            cloned_updaters = []
            for updater in mobject.updaters:
                # Make the cloned updater use the cloned Mobjects as free variables
                # rather than the original ones. Analyzing function bytecode with the
                # dis module will help in understanding this.
                # https://docs.python.org/3/library/dis.html
                # TODO: Do the same for function calls recursively.
                free_variable_map = inspect.getclosurevars(updater).nonlocals
                cloned_co_freevars = []
                cloned_closure = []
                for free_variable_name in updater.__code__.co_freevars:
                    free_variable_value = free_variable_map[free_variable_name]

                    # If the referenced variable has not been cloned, raise.
                    if id(free_variable_value) not in clone_from_id:
                        raise Exception(
                            f"{free_variable_name} is referenced from an updater "
                            "but is not an attribute of the Scene, which isn't "
                            "allowed.",
                        )

                    # Add the cloned object's name to the free variable list.
                    cloned_co_freevars.append(free_variable_name)

                    # Add a cell containing the cloned object's reference to the
                    # closure list.
                    cloned_closure.append(
                        types.CellType(clone_from_id[id(free_variable_value)]),
                    )

                cloned_updater = types.FunctionType(
                    updater.__code__.replace(co_freevars=tuple(cloned_co_freevars)),
                    updater.__globals__,
                    updater.__name__,
                    updater.__defaults__,
                    tuple(cloned_closure),
                )
                cloned_updaters.append(cloned_updater)
            mobject_clone = clone_from_id[id(mobject)]
            mobject_clone.updaters = cloned_updaters
            if len(cloned_updaters) > 0:
                result.mobject_updater_lists.append((mobject_clone, cloned_updaters))
        return result
    
    def init_element(self, element, event_name):
        self.camera.init_element(element, event_name)
        self.element.addEventListener(
            "mousedown",
            create_proxy(self.on_mouse_down),
        )
        self.element.addEventListener(
            "mouseup",
            create_proxy(self.on_mouse_up),
        )
        self.element.addEventListener(
            "mousemove",
            create_proxy(self.on_mouse_move),
        )
        self.element.addEventListener(
            "click",
            create_proxy(self.on_mouse_click),
        )
        self.element.addEventListener(
            "keydown",
            create_proxy(self.on_key_down),
        )
        self.element.addEventListener(
            "keyup",
            create_proxy(self.on_key_up),
        )

    def render_frame(self) -> None:
        self.renderer.render(self, self.time, self.moving_mobjects)
    
    @property
    def element(self):
        return self.camera.element

    async def render(self, preview: bool = False):
        """
        Renders this Scene.

        Parameters
        ---------
        preview
            If true, opens scene in a file viewer.
        """
        await self.setup()
        try:
            await self.construct()
        except EndSceneEarlyException:
            pass
        except RerunSceneException:
            self.remove(*self.mobjects)
            self.renderer.clear_screen()
            self.renderer.num_plays = 0
            return True
        await self.tear_down()
        # We have to reset these settings in case of multiple renders.
        self.renderer.scene_finished(self)

        # Show info only if animations are rendered or to get image
        if (
            self.renderer.num_plays
            or config["format"] == "png"
            or config["save_last_frame"]
        ):
            logger.info(
                f"Rendered {str(self)}\nPlayed {self.renderer.num_plays} animations",
            )

        # If preview open up the render after rendering.
        if preview:
            config["preview"] = True
    
    async def on_mouse_click(
        self,
        event,
    ) -> None:
        """
        Called when a mouse click event occurs.

        Parameters
        ----------
        event
            The mouse click event.
        """
        pass
    
    async def on_mouse_down(
        self,
        event,
    ) -> None:
        pass

    async def on_mouse_up(
        self,
        event,
    ) -> None:
        pass

    async def on_mouse_move(
        self,
        event,
    ) -> None:
        pass

    async def on_key_down(
        self,
        event,
    ) -> None:
        pass

    async def on_key_up(
        self,
        event,
    ) -> None:
        pass

    async def setup(self):
        """
        This is meant to be implemented by any scenes which
        are commonly subclassed, and have some common setup
        involved before the construct method is called.
        """
        pass

    async def tear_down(self):
        """
        This is meant to be implemented by any scenes which
        are commonly subclassed, and have some common method
        to be invoked before the scene ends.
        """
        pass

    async def construct(self):
        """Add content to the Scene.

        From within :meth:`Scene.construct`, display mobjects on screen by calling
        :meth:`Scene.add` and remove them from screen by calling :meth:`Scene.remove`.
        All mobjects currently on screen are kept in :attr:`Scene.mobjects`.  Play
        animations by calling :meth:`Scene.play`.

        Notes
        -----
        Initialization code should go in :meth:`Scene.setup`.  Termination code should
        go in :meth:`Scene.tear_down`.

        Examples
        --------
        A typical manim script includes a class derived from :class:`Scene` with an
        overridden :meth:`Scene.construct` method:

        .. code-block:: python

            class MyScene(Scene):
                def construct(self):
                    self.play(Write(Text("Hello World!")))

        See Also
        --------
        :meth:`Scene.setup`
        :meth:`Scene.render`
        :meth:`Scene.tear_down`

        """
        pass  # To be implemented in subclasses

    def __str__(self):
        return self.__class__.__name__

    def get_attrs(self, *keys: str):
        """
        Gets attributes of a scene given the attribute's identifier/name.

        Parameters
        ----------
        *keys
            Name(s) of the argument(s) to return the attribute of.

        Returns
        -------
        list
            List of attributes of the passed identifiers.
        """
        return [getattr(self, key) for key in keys]

    def update_mobjects(self, dt: float):
        """
        Begins updating all mobjects in the Scene.

        Parameters
        ----------
        dt
            Change in time between updates. Defaults (mostly) to 1/frames_per_second
        """
        for mobject in self.mobjects:
            mobject.update(dt)

    def update_meshes(self, dt):
        for obj in self.meshes:
            for mesh in obj.get_family():
                mesh.update(dt)

    def update_self(self, dt: float):
        """Run all scene updater functions.

        Among all types of update functions (mobject updaters, mesh updaters,
        scene updaters), scene update functions are called last.

        Parameters
        ----------
        dt
            Scene time since last update.

        See Also
        --------
        :meth:`.Scene.add_updater`
        :meth:`.Scene.remove_updater`
        """
        for func in self.updaters:
            func(dt)

    def should_update_mobjects(self) -> bool:
        """
        Returns True if the mobjects of this scene should be updated.

        In particular, this checks whether

        - the :attr:`always_update_mobjects` attribute of :class:`.Scene`
          is set to ``True``,
        - the :class:`.Scene` itself has time-based updaters attached,
        - any mobject in this :class:`.Scene` has time-based updaters attached.

        This is only called when a single Wait animation is played.
        """
        wait_animation = self.animations[0]
        if wait_animation.is_static_wait is None:
            should_update = (
                self.always_update_mobjects
                or self.updaters
                or wait_animation.stop_condition is not None
                or any(
                    mob.has_time_based_updater()
                    for mob in self.get_mobject_family_members()
                )
            )
            wait_animation.is_static_wait = not should_update
        return not wait_animation.is_static_wait

    def get_top_level_mobjects(self):
        """
        Returns all mobjects which are not submobjects.

        Returns
        -------
        list
            List of top level mobjects.
        """
        # Return only those which are not in the family
        # of another mobject from the scene
        families = [m.get_family() for m in self.mobjects]

        def is_top_level(mobject):
            num_families = sum((mobject in family) for family in families)
            return num_families == 1

        return list(filter(is_top_level, self.mobjects))

    def get_mobject_family_members(self):
        """
        Returns list of family-members of all mobjects in scene.
        If a Circle() and a VGroup(Rectangle(),Triangle()) were added,
        it returns not only the Circle(), Rectangle() and Triangle(), but
        also the VGroup() object.

        Returns
        -------
        list
            List of mobject family members.
        """
        if config.renderer == RendererType.CANVAS:
            return extract_mobject_family_members(
                self.mobjects,
                use_z_index=self.renderer.camera.use_z_index,
            )
        raise NotImplementedError(
            "get_mobject_family_members is not implemented for the active renderer."
        )

    def add(self, *mobjects: Mobject):
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.

        Parameters
        ---------
        *mobjects
            Mobjects to add.

        Returns
        -------
        Scene
            The same scene after adding the Mobjects in.

        """
        if config.renderer == RendererType.CANVAS:
            mobjects = [*mobjects, *self.foreground_mobjects]
            self.restructure_mobjects(to_remove=mobjects)
            self.mobjects += mobjects
            if self.moving_mobjects:
                self.restructure_mobjects(
                    to_remove=mobjects,
                    mobject_list_name="moving_mobjects",
                )
                self.moving_mobjects += mobjects
        return self

    def add_mobjects_from_animations(self, animations: list[Animation]) -> None:
        curr_mobjects = self.get_mobject_family_members()
        for animation in animations:
            if animation.is_introducer():
                continue
            # Anything animated that's not already in the
            # scene gets added to the scene
            mob = animation.mobject
            if mob is not None and mob not in curr_mobjects:
                self.add(mob)
                curr_mobjects += mob.get_family()

    def remove(self, *mobjects: Mobject):
        """
        Removes mobjects in the passed list of mobjects
        from the scene and the foreground, by removing them
        from "mobjects" and "foreground_mobjects"

        Parameters
        ----------
        *mobjects
            The mobjects to remove.
        """
        if config.renderer == RendererType.CANVAS:
            for list_name in "mobjects", "foreground_mobjects":
                self.restructure_mobjects(mobjects, list_name, False)
            return self

    def replace(self, old_mobject: Mobject, new_mobject: Mobject) -> None:
        """Replace one mobject in the scene with another, preserving draw order.

        If ``old_mobject`` is a submobject of some other Mobject (e.g. a
        :class:`.Group`), the new_mobject will replace it inside the group,
        without otherwise changing the parent mobject.

        Parameters
        ----------
        old_mobject
            The mobject to be replaced. Must be present in the scene.
        new_mobject
            A mobject which must not already be in the scene.

        """
        if old_mobject is None or new_mobject is None:
            raise ValueError("Specified mobjects cannot be None")

        def replace_in_list(
            mobj_list: list[Mobject], old_m: Mobject, new_m: Mobject
        ) -> bool:
            # We use breadth-first search because some Mobjects get very deep and
            # we expect top-level elements to be the most common targets for replace.
            for i in range(0, len(mobj_list)):
                # Is this the old mobject?
                if mobj_list[i] == old_m:
                    # If so, write the new object to the same spot and stop looking.
                    mobj_list[i] = new_m
                    return True
            # Now check all the children of all these mobs.
            for mob in mobj_list:  # noqa: SIM110
                if replace_in_list(mob.submobjects, old_m, new_m):
                    # If we found it in a submobject, stop looking.
                    return True
            # If we did not find the mobject in the mobject list or any submobjects,
            # (or the list was empty), indicate we did not make the replacement.
            return False

        # Make use of short-circuiting conditionals to check mobjects and then
        # foreground_mobjects
        replaced = replace_in_list(
            self.mobjects, old_mobject, new_mobject
        ) or replace_in_list(self.foreground_mobjects, old_mobject, new_mobject)

        if not replaced:
            raise ValueError(f"Could not find {old_mobject} in scene")

    def add_updater(self, func: Callable[[float], None]) -> None:
        """Add an update function to the scene.

        The scene updater functions are run every frame,
        and they are the last type of updaters to run.

        .. WARNING::

            When using the Cairo renderer, scene updaters that
            modify mobjects are not detected in the same way
            that mobject updaters are. To be more concrete,
            a mobject only modified via a scene updater will
            not necessarily be added to the list of *moving
            mobjects* and thus might not be updated every frame.

            TL;DR: Use mobject updaters to update mobjects.

        Parameters
        ----------
        func
            The updater function. It takes a float, which is the
            time difference since the last update (usually equal
            to the frame rate).

        See also
        --------
        :meth:`.Scene.remove_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters.append(func)

    def remove_updater(self, func: Callable[[float], None]) -> None:
        """Remove an update function from the scene.

        Parameters
        ----------
        func
            The updater function to be removed.

        See also
        --------
        :meth:`.Scene.add_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters = [f for f in self.updaters if f is not func]

    def restructure_mobjects(
        self,
        to_remove: Sequence[Mobject],
        mobject_list_name: str = "mobjects",
        extract_families: bool = True,
    ):
        """
        tl:wr
            If your scene has a Group(), and you removed a mobject from the Group,
            this dissolves the group and puts the rest of the mobjects directly
            in self.mobjects or self.foreground_mobjects.

        In cases where the scene contains a group, e.g. Group(m1, m2, m3), but one
        of its submobjects is removed, e.g. scene.remove(m1), the list of mobjects
        will be edited to contain other submobjects, but not m1, e.g. it will now
        insert m2 and m3 to where the group once was.

        Parameters
        ----------
        to_remove
            The Mobject to remove.

        mobject_list_name
            The list of mobjects ("mobjects", "foreground_mobjects" etc) to remove from.

        extract_families
            Whether the mobject's families should be recursively extracted.

        Returns
        -------
        Scene
            The Scene mobject with restructured Mobjects.
        """
        if extract_families:
            to_remove = extract_mobject_family_members(
                to_remove,
                use_z_index=self.renderer.camera.use_z_index,
            )
        _list = getattr(self, mobject_list_name)
        new_list = self.get_restructured_mobject_list(_list, to_remove)
        setattr(self, mobject_list_name, new_list)
        return self

    def get_restructured_mobject_list(self, mobjects: list, to_remove: list):
        """
        Given a list of mobjects and a list of mobjects to be removed, this
        filters out the removable mobjects from the list of mobjects.

        Parameters
        ----------

        mobjects
            The Mobjects to check.

        to_remove
            The list of mobjects to remove.

        Returns
        -------
        list
            The list of mobjects with the mobjects to remove removed.
        """
        new_mobjects = []

        def add_safe_mobjects_from_list(list_to_examine, set_to_remove):
            for mob in list_to_examine:
                if mob in set_to_remove:
                    continue
                intersect = set_to_remove.intersection(mob.get_family())
                if intersect:
                    add_safe_mobjects_from_list(mob.submobjects, intersect)
                else:
                    new_mobjects.append(mob)

        add_safe_mobjects_from_list(mobjects, set(to_remove))
        return new_mobjects

    # TODO, remove this, and calls to this
    def add_foreground_mobjects(self, *mobjects: Mobject):
        """
        Adds mobjects to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        *mobjects
            The Mobjects to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects added.
        """
        self.foreground_mobjects = list_update(self.foreground_mobjects, mobjects)
        self.add(*mobjects)
        return self

    def add_foreground_mobject(self, mobject: Mobject):
        """
        Adds a single mobject to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        mobject
            The Mobject to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject added.
        """
        return self.add_foreground_mobjects(mobject)

    def remove_foreground_mobjects(self, *to_remove: Mobject):
        """
        Removes mobjects from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        *to_remove
            The mobject(s) to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects removed.
        """
        self.restructure_mobjects(to_remove, "foreground_mobjects")
        return self

    def remove_foreground_mobject(self, mobject: Mobject):
        """
        Removes a single mobject from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        mobject
            The mobject to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject removed.
        """
        return self.remove_foreground_mobjects(mobject)

    def bring_to_front(self, *mobjects: Mobject):
        """
        Adds the passed mobjects to the scene again,
        pushing them to he front of the scene.

        Parameters
        ----------
        *mobjects
            The mobject(s) to bring to the front of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects brought to the front
            of the scene.
        """
        self.add(*mobjects)
        return self

    def bring_to_back(self, *mobjects: Mobject):
        """
        Removes the mobject from the scene and
        adds them to the back of the scene.

        Parameters
        ----------
        *mobjects
            The mobject(s) to push to the back of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects pushed to the back
            of the scene.
        """
        self.remove(*mobjects)
        self.mobjects = list(mobjects) + self.mobjects
        return self

    def clear(self):
        """
        Removes all mobjects present in self.mobjects
        and self.foreground_mobjects from the scene.

        Returns
        ------
        Scene
            The Scene, with all of its mobjects in
            self.mobjects and self.foreground_mobjects
            removed.
        """
        self.mobjects = []
        self.foreground_mobjects = []
        return self

    def get_moving_mobjects(self, *animations: Animation):
        """
        Gets all moving mobjects in the passed animation(s).

        Parameters
        ----------
        *animations
            The animations to check for moving mobjects.

        Returns
        ------
        list
            The list of mobjects that could be moving in
            the Animation(s)
        """
        # Go through mobjects from start to end, and
        # as soon as there's one that needs updating of
        # some kind per frame, return the list from that
        # point forward.
        animation_mobjects = [anim.mobject for anim in animations]
        mobjects = self.get_mobject_family_members()
        for i, mob in enumerate(mobjects):
            update_possibilities = [
                mob in animation_mobjects,
                len(mob.get_family_updaters()) > 0,
                mob in self.foreground_mobjects,
            ]
            if any(update_possibilities):
                return mobjects[i:]
        return []

    def get_moving_and_static_mobjects(self, animations):
        all_mobjects = list_update(self.mobjects, self.foreground_mobjects)
        all_mobject_families = extract_mobject_family_members(
            all_mobjects,
            use_z_index=self.renderer.camera.use_z_index,
            only_those_with_points=True,
        )
        moving_mobjects = self.get_moving_mobjects(*animations)
        all_moving_mobject_families = extract_mobject_family_members(
            moving_mobjects,
            use_z_index=self.renderer.camera.use_z_index,
        )
        static_mobjects = list_difference_update(
            all_mobject_families,
            all_moving_mobject_families,
        )
        return all_moving_mobject_families, static_mobjects

    def compile_animations(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs,
    ):
        """
        Creates _MethodAnimations from any _AnimationBuilders and updates animation
        kwargs with kwargs passed to play().

        Parameters
        ----------
        *args
            Animations to be played.
        **kwargs
            Configuration for the call to play().

        Returns
        -------
        Tuple[:class:`Animation`]
            Animations to be played.
        """
        animations = []
        arg_anims = flatten_iterable_parameters(args)
        # Allow passing a generator to self.play instead of comma separated arguments
        for arg in arg_anims:
            try:
                animations.append(prepare_animation(arg))
            except TypeError as e:
                if inspect.ismethod(arg):
                    raise TypeError(
                        "Passing Mobject methods to Scene.play is no longer"
                        " supported. Use Mobject.animate instead.",
                    ) from e
                else:
                    raise TypeError(
                        f"Unexpected argument {arg} passed to Scene.play().",
                    ) from e

        for animation in animations:
            for k, v in kwargs.items():
                setattr(animation, k, v)

        return animations

    @classmethod
    def validate_run_time(
        cls,
        run_time: float,
        method: Callable[[Any, ...], Any],
        parameter_name: str = "run_time",
    ) -> float:
        method_name = f"{cls.__name__}.{method.__name__}()"
        if run_time <= 0:
            raise ValueError(
                f"{method_name} has a {parameter_name} of "
                f"{run_time:g} <= 0 seconds which Manim cannot render. "
                f"The {parameter_name} must be a positive number."
            )

        # config.frame_rate holds the number of frames per second
        fps = config.frame_rate
        seconds_per_frame = 1 / fps
        if run_time < seconds_per_frame:
            logger.warning(
                f"The original {parameter_name} of {method_name}, "
                f"{run_time:g} seconds, is too short for the current frame "
                f"rate of {fps:g} FPS. Rendering with the shortest possible "
                f"{parameter_name} of {seconds_per_frame:g} seconds instead."
            )
            run_time = seconds_per_frame

        return run_time

    def get_run_time(self, animations: list[Animation]):
        """
        Gets the total run time for a list of animations.

        Parameters
        ----------
        animations
            A list of the animations whose total
            ``run_time`` is to be calculated.

        Returns
        -------
        float
            The total ``run_time`` of all of the animations in the list.
        """
        run_time = max(animation.run_time for animation in animations)
        run_time = self.validate_run_time(run_time, self.play, "total run_time")
        return run_time

    async def play(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        subcaption=None,
        subcaption_duration=None,
        subcaption_offset=0,
        **kwargs,
    ):
        r"""Plays an animation in this scene.

        Parameters
        ----------

        args
            Animations to be played.
        subcaption
            The content of the external subcaption that should
            be added during the animation.
        subcaption_duration
            The duration for which the specified subcaption is
            added. If ``None`` (the default), the run time of the
            animation is taken.
        subcaption_offset
            An offset (in seconds) for the start time of the
            added subcaption.
        kwargs
            All other keywords are passed to the renderer.

        """

        start_time = self.time
        await self.renderer.play(self, *args, **kwargs)
        run_time = self.time - start_time
        if subcaption:
            if subcaption_duration is None:
                subcaption_duration = run_time
            # The start of the subcaption needs to be offset by the
            # run_time of the animation because it is added after
            # the animation has already been played (and Scene.time
            # has already been updated).
            self.add_subcaption(
                content=subcaption,
                duration=subcaption_duration,
                offset=-run_time + subcaption_offset,
            )

    async def wait(
        self,
        duration: float = DEFAULT_WAIT_TIME,
        stop_condition: Callable[[], bool] | None = None,
        frozen_frame: bool | None = None,
    ):
        """Plays a "no operation" animation.

        Parameters
        ----------
        duration
            The run time of the animation.
        stop_condition
            A function without positional arguments that is evaluated every time
            a frame is rendered. The animation only stops when the return value
            of the function is truthy, or when the time specified in ``duration``
            passes.
        frozen_frame
            If True, updater functions are not evaluated, and the animation outputs
            a frozen frame. If False, updater functions are called and frames
            are rendered as usual. If None (the default), the scene tries to
            determine whether or not the frame is frozen on its own.

        See also
        --------
        :class:`.Wait`, :meth:`.should_mobjects_update`
        """
        duration = self.validate_run_time(duration, self.wait, "duration")
        await self.play(
            Wait(
                run_time=duration,
                stop_condition=stop_condition,
                frozen_frame=frozen_frame,
            )
        )

    async def pause(self, duration: float = DEFAULT_WAIT_TIME):
        """Pauses the scene (i.e., displays a frozen frame).

        This is an alias for :meth:`.wait` with ``frozen_frame``
        set to ``True``.

        Parameters
        ----------
        duration
            The duration of the pause.

        See also
        --------
        :meth:`.wait`, :class:`.Wait`
        """
        duration = self.validate_run_time(duration, self.pause, "duration")
        await self.wait(duration=duration, frozen_frame=True)

    async def wait_until(self, stop_condition: Callable[[], bool], max_time: float = 60):
        """Wait until a condition is satisfied, up to a given maximum duration.

        Parameters
        ----------
        stop_condition
            A function with no arguments that determines whether or not the
            scene should keep waiting.
        max_time
            The maximum wait time in seconds.
        """
        max_time = self.validate_run_time(max_time, self.wait_until, "max_time")
        await self.wait(max_time, stop_condition=stop_condition)

    def compile_animation_data(
        self,
        *animations: Animation | Mobject | _AnimationBuilder,
        **play_kwargs,
    ):
        """Given a list of animations, compile the corresponding
        static and moving mobjects, and gather the animation durations.

        This also begins the animations.

        Parameters
        ----------
        animations
            Animation or mobject with mobject method and params
        play_kwargs
            Named parameters affecting what was passed in ``animations``,
            e.g. ``run_time``, ``lag_ratio`` and so on.

        Returns
        -------
        self, None
            None if there is nothing to play, or self otherwise.
        """
        # NOTE TODO : returns statement of this method are wrong. It should return nothing, as it makes a little sense to get any information from this method.
        # The return are kept to keep webgl renderer from breaking.
        if len(animations) == 0:
            raise ValueError("Called Scene.play with no animations")

        self.animations = self.compile_animations(*animations, **play_kwargs)
        self.add_mobjects_from_animations(self.animations)

        self.last_t = 0
        self.stop_condition = None
        self.moving_mobjects = []
        self.static_mobjects = []

        self.duration = self.get_run_time(self.animations)
        if len(self.animations) == 1 and isinstance(self.animations[0], Wait):
            if self.should_update_mobjects():
                self.update_mobjects(dt=0)  # Any problems with this?
                self.stop_condition = self.animations[0].stop_condition
            else:
                # Static image logic when the wait is static is done by the renderer, not here.
                self.animations[0].is_static_wait = True
                return None

        return self

    def begin_animations(self) -> None:
        """Start the animations of the scene."""
        for animation in self.animations:
            animation._setup_scene(self)
            animation.begin()

        if config.renderer == RendererType.CANVAS:
            # Paint all non-moving objects onto the screen, so they don't
            # have to be rendered every frame
            (
                self.moving_mobjects,
                self.static_mobjects,
            ) = self.get_moving_and_static_mobjects(self.animations)
    
    @property
    def static_mobjects(self) -> list[Mobject]:
        return []
    
    @static_mobjects.setter
    def static_mobjects(self, value: list[Mobject]):
        pass

    @property
    def moving_mobjects(self) -> list[Mobject]:
        """The list of mobjects that are currently moving in the scene."""
        return self.mobjects
    
    @moving_mobjects.setter
    def moving_mobjects(self, value: list[Mobject]):
        """Set the list of mobjects that are currently moving in the scene."""
        pass

    def is_current_animation_frozen_frame(self) -> bool:
        """Returns whether the current animation produces a static frame (generally a Wait)."""
        return (
            isinstance(self.animations[0], Wait)
            and len(self.animations) == 1
            and self.animations[0].is_static_wait
        )

    async def play_internal(self, skip_rendering: bool = False):
        """
        This method is used to prep the animations for rendering,
        apply the arguments and parameters required to them,
        render them, and write them to the video file.

        Parameters
        ----------
        skip_rendering
            Whether the rendering should be skipped, by default False
        """
        self.duration = self.get_run_time(self.animations)
        for t in np.arange(0, self.duration, 1 / self.camera.frame_rate):
            task = asyncio.create_task(asyncio.sleep(1 / self.camera.frame_rate))
            self.update_to_time(t)
            if not skip_rendering and not self.skip_animation_preview:
                self.renderer.render(self, t, self.moving_mobjects)
            if self.stop_condition is not None and self.stop_condition():
                break
            await task

        for animation in self.animations:
            animation.finish()
            animation.clean_up_from_scene(self)
        if not self.renderer.skip_animations:
            self.update_mobjects(0)
        self.renderer.static_image = None

    def update_to_time(self, t):
        dt = t - self.last_t
        self.last_t = t
        for animation in self.animations:
            animation.update_mobjects(dt)
            alpha = t / animation.run_time
            animation.interpolate(alpha)
        self.update_mobjects(dt)
        self.update_meshes(dt)
        self.update_self(dt)
