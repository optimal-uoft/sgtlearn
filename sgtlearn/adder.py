from example import Adder as _Adder


class Adder:
    """Thin Python wrapper; computation runs in the compiled extension."""

    def __init__(self) -> None:
        self._inner = _Adder()



    def add(self, x: float, y: float) -> float:
        return self._inner.add(x, y)
