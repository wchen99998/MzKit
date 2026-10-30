"""
Qt signals when PyQt5 is installed, else a plain-Python stand-in with the
same connect/disconnect/emit API, so the core library runs headless.
"""

try:
    from PyQt5.QtCore import QObject, pyqtSignal
except ImportError:  # no `gui` dependency group
    class QObject:
        def __init__(self, *args, **kwargs):
            pass

    class _BoundSignal:
        def __init__(self):
            self._slots = []

        def connect(self, slot):
            self._slots.append(slot)

        def disconnect(self, slot=None):
            if slot is None:
                self._slots.clear()
            else:
                self._slots.remove(slot)

        def emit(self, *args):
            for slot in list(self._slots):
                slot(*args)

    class pyqtSignal:
        """Class attribute giving each instance its own signal, like Qt's."""
        def __init__(self, *types, **kwargs):
            self._attr = None

        def __set_name__(self, owner, name):
            self._attr = f'_signal_{name}'

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            bound = obj.__dict__.get(self._attr)
            if bound is None:
                bound = obj.__dict__[self._attr] = _BoundSignal()
            return bound
