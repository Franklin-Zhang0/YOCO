"""Glove keypoint publishing and refinement helpers."""

__all__ = ["GloveKeypointSample", "GloveRefinePublisher"]


def __getattr__(name: str):
    if name in __all__:
        from .glove_refine_publisher import GloveKeypointSample, GloveRefinePublisher

        return {
            "GloveKeypointSample": GloveKeypointSample,
            "GloveRefinePublisher": GloveRefinePublisher,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
