"""The metric graph both graphing tabs draw into, and the pieces it is made of.

:class:`~.metric_graph.MetricGraph` is the widget the Batch average and Compare
tabs share — the same implementation used twice, which is what keeps their
framing, level weighting, and point readout from drifting apart. It is composed
of four collaborators kept in their own modules so the widget stays about
*drawing*:

* :mod:`.palette` — the colour cycle the graph and the Compare tab's swatches
  both spend, so a legend key always matches its curve.
* :mod:`.framing` — the framing buttons and their enable state. What each one
  frames stays with the graph, which owns the bounds.
* :mod:`.axis_bounds` — the button and form for typing exact axis bounds, for
  the frames the snapping buttons cannot express. Applying them stays with the
  graph, which owns the plot.
* :mod:`.readout` — the click-to-read box and its pick marker. Which sample was
  clicked stays with the graph, which owns the drawn series.
"""

from __future__ import annotations

from .metric_graph import MetricGraph
from .palette import _color_swatch, _MARK_COLOR, _MUTED_INK, _SERIES_COLORS, series_color

__all__ = [
    "MetricGraph",
    "series_color",
    "_color_swatch",
    "_MARK_COLOR",
    "_MUTED_INK",
    "_SERIES_COLORS",
]
