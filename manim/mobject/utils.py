"""Utilities for working with mobjects."""

from __future__ import annotations

__all__ = [
    "get_mobject_class",
    "get_vectorized_mobject_class",
]

from .._config import config
from ..constants import RendererType
from .mobject import Mobject
from .types.vectorized_mobject import VMobject


def get_mobject_class() -> type:
    """Gets the base mobject class, depending on the currently active renderer.

    .. NOTE::

        This method is intended to be used in the code base of Manim itself
        or in plugins where code should work independent of the selected
        renderer.

    Examples
    --------

    The function has to be explicitly imported. We test that
    the name of the returned class is one of the known mobject
    base classes::

        >>> from manim.mobject.utils import get_mobject_class
        >>> get_mobject_class().__name__ in ['Mobject', 'OpenGLMobject']
        True
    """
    if config.renderer == RendererType.CANVAS:
        return Mobject
    raise NotImplementedError(
        "Base mobjects are not implemented for the active renderer."
    )


def get_vectorized_mobject_class() -> type:
    """Gets the vectorized mobject class, depending on the currently
    active renderer.

    .. NOTE::

        This method is intended to be used in the code base of Manim itself
        or in plugins where code should work independent of the selected
        renderer.

    Examples
    --------

    The function has to be explicitly imported. We test that
    the name of the returned class is one of the known mobject
    base classes::

        >>> from manim.mobject.utils import get_vectorized_mobject_class
        >>> get_vectorized_mobject_class().__name__ in ['VMobject', 'OpenGLVMobject']
        True
    """
    if config.renderer == RendererType.CANVAS:
        return VMobject
