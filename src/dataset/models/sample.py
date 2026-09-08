"""One feature and the observations of it a read drew, which is what a run trains on."""

from __future__ import annotations

from dataclasses import dataclass

from building.metadata.observation import ObservationMetadata
from shared.models.feature import Feature


@dataclass(frozen=True, slots=True)
class Sample:
    """A geological feature and the observations of it one read drew.

    Attributes:
        feature: The feature every one of them was cropped to.
        observations: What was drawn of it, at most as many of each instrument
            as the config asks for.
    """

    feature: Feature
    observations: tuple[ObservationMetadata, ...]

    @property
    def identity(self) -> tuple[str, str]:
        """Return what tells this feature from every other.

        Returns:
            identity: Its class and its name, as the catalogue spells them.
        """
        return (self.feature.feature_class, self.feature.feature_name)

    @property
    def instruments(self) -> tuple[str, ...]:
        """Return which instruments the draw reached, in the order they were drawn.

        Returns:
            instruments: Each instrument once, as ODE names it.
        """
        return tuple(dict.fromkeys(one.instrument for one in self.observations))
