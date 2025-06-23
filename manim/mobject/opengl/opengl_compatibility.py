from __future__ import annotations

from abc import ABCMeta

__all__ = ["ConvertToOpenGL"]


class ConvertToOpenGL(ABCMeta):
    """Metaclass for swapping (V)Mobject with its OpenGL counterpart at runtime
    depending on config.renderer. This metaclass should only need to be inherited
    on the lowest order inheritance classes such as Mobject and VMobject.
    """

    _converted_classes = []

    def __new__(mcls, name, bases, namespace):
        return super().__new__(mcls, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)