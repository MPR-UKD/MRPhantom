"""Abstract base class for all MR phantom types.

All concrete phantom classes (e.g. :class:`~mrphantom.dwi.DWIPhantom`) inherit
from :class:`MRPhantom` and must implement :meth:`load_case` and
:meth:`synthesize`.

Design conventions (sklearn-inspired, domain-adapted)
------------------------------------------------------
- Constructor receives *hyperparameters only* (acquisition settings such as
  ``snr`` and ``seed``).  No case data is accepted at construction time.
- :meth:`load_case` attaches a case (segmentation + diffusion parameters) and
  returns ``self`` to allow method chaining.
- :meth:`synthesize` runs the signal model and returns ``self``; results are
  exposed via read-only properties on the concrete class.
- :meth:`get_params` / :meth:`set_params` provide consistent introspection,
  mirroring the sklearn estimator protocol.
- :meth:`__repr__` shows the class name and all non-``None`` parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MRPhantom(ABC):
    """Abstract base class for MR phantom synthesizers.

    Attributes:
        snr: Signal-to-noise ratio for Rician noise simulation. ``None``
            (default) for a noise-free phantom.
        seed: Integer random seed for reproducible noise. Ignored when ``snr``
            is ``None``.
    """

    def __init__(self, *, snr: float | None = None, seed: int | None = None) -> None:
        """Initialise with acquisition parameters only."""
        self.snr = snr
        self.seed = seed

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_case(self, config) -> "MRPhantom":
        """Attach case data (segmentation + diffusion parameters).

        Args:
            config: A case configuration object understood by the concrete
                subclass (e.g. :class:`~mrphantom.config.PhantomConfig`).

        Returns:
            MRPhantom: self, allows chaining:
            ``phantom.load_case(config).synthesize()``.
        """

    @abstractmethod
    def synthesize(self) -> "MRPhantom":
        """Run the signal synthesis model.

        Must be called after :meth:`load_case`. Results are stored on the
        instance and accessible via the concrete class's read-only properties.

        Returns:
            MRPhantom: self, allows chaining:
            ``phantom.load_case(config).synthesize().save(...)``.
        """

    # ------------------------------------------------------------------
    # Introspection — sklearn-style
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return constructor parameters as a dictionary.

        Returns:
            dict: ``{"snr": ..., "seed": ...}`` (and any additional parameters
            defined by concrete subclasses).

        Examples:
            >>> phantom.get_params()
            {'snr': 30.0, 'seed': 42}
        """
        return {"snr": self.snr, "seed": self.seed}

    def set_params(self, **params) -> "MRPhantom":
        """Set one or more parameters and return ``self``.

        Args:
            **params: Keyword arguments matching constructor parameter names.

        Returns:
            MRPhantom: self, allows chaining:
            ``phantom.set_params(snr=50).load_case(config)``.

        Raises:
            ValueError: If an unknown parameter name is passed.

        Examples:
            >>> phantom.set_params(snr=50, seed=0)
        """
        valid = set(self.get_params())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"{type(self).__name__}.set_params() received unknown "
                    f"parameter {key!r}.  Valid parameters: {sorted(valid)}."
                )
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        params = self.get_params()
        non_default = {k: v for k, v in params.items() if v is not None}
        args = ", ".join(f"{k}={v!r}" for k, v in non_default.items())
        return f"{type(self).__name__}({args})"
