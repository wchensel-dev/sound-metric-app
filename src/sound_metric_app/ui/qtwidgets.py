"""Qt-generic widget helpers shared across the workflow window's views.

Everything here is about *widgets*, not about shots, batches, or metrics: tree
styling and grid lines, the expansion snapshot both archive trees restore across
a rebuild, the button flash that confirms an invisible action, and the SKU-filter
dropdown the data-bank and batch-average tabs both front their tree with.

Kept apart from the views so a rendering rule that must read the same on two tabs
is written once. Nothing here imports a view or the controller.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

#: Sentinel row in a SKU-filter dropdown that clears the filter (data ``None``).
_ALL_SKUS_LABEL = "All SKUs"

#: How long a flashed confirmation holds its text before reverting (see
#: :func:`_flash_button`).
_COPIED_FLASH_MS = 1200

#: Tint behind the Batch average tree's slot rows, so an averaged row is legible
#: at a glance against the individual shots nested under it. Deliberately
#: low-alpha: it composites over whichever base/alternate-base colour the active
#: theme paints, so the same amber reads as a soft wash in light and dark alike.
_AVERAGE_ROW_TINT = QtGui.QColor(255, 176, 32, 64)


class _TopLevelRowTint(QtWidgets.QStyledItemDelegate):
    """Wash a colour behind a tree's top-level rows, leaving children plain.

    ``QTreeWidgetItem.setBackground`` cannot do this job here: once a stylesheet
    applies to the view (see ``_style_grid_tree``), Qt paints the row background
    from the stylesheet and drops the item's own brush, so the tint never
    appears. Painting it ourselves sidesteps that. The fill goes *under*
    ``super().paint()``, so the stylesheet's grid lines, the selection
    highlight, and the text all still draw over it at full strength.
    """

    def __init__(self, color: QtGui.QColor, parent=None):
        super().__init__(parent)
        self._color = color

    def paint(self, painter, option, index) -> None:
        if not index.parent().isValid():
            painter.fillRect(option.rect, self._color)
        super().paint(painter, option, index)


def _repopulate_sku_filter(combo: QtWidgets.QComboBox, skus: list[str]) -> None:
    """Refill a SKU-filter dropdown, preserving the current selection.

    Row 0 is the ``All SKUs`` sentinel (data ``None``, meaning "no filter"); each
    remaining row carries a SKU string as both its label and data. Signals are
    blocked over the rebuild so it does not fire the combo's
    ``currentIndexChanged`` — the caller re-reads the selection and refreshes
    explicitly. A previously selected SKU that no longer exists (its last
    combination was swept) falls back to ``All SKUs``.
    """
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(_ALL_SKUS_LABEL, None)
    for sku in skus:
        combo.addItem(sku, sku)
    index = combo.findData(current)
    combo.setCurrentIndex(index if index >= 0 else 0)
    combo.blockSignals(False)


def _style_grid_tree(tree: QtWidgets.QTreeWidget) -> None:
    """Give a ``QTreeWidget`` visible column/row grid lines.

    A tree has no built-in grid, so we draw one: per-item borders supply the
    column and row rules and alternating row colours make wide numeric rows
    easier to scan. Colours come from ``palette(...)`` so the grid tracks the
    active light/dark theme instead of clashing with it. Shared by the Report
    and Batches trees so both read the same way.
    """
    tree.setAlternatingRowColors(True)
    tree.header().setSectionsMovable(False)
    tree.setStyleSheet(
        "QTreeWidget {"
        " alternate-background-color: palette(alternate-base);"
        " background: palette(base); }"
        "QTreeWidget::item {"
        " border-right: 1px solid palette(mid);"
        " border-bottom: 1px solid palette(mid);"
        " padding: 2px 4px; }"
        "QTreeWidget::item:selected {"
        " background: palette(highlight);"
        " color: palette(highlighted-text); }"
    )


def _flash_button(button: QtWidgets.QPushButton, message: str, revert_to: str) -> None:
    """Briefly swap a button's text to confirm an action that leaves no mark.

    Both of the Batch average tree's row buttons act somewhere the operator is
    not looking — the clipboard, and the Compare tab — so the button itself has
    to acknowledge the click. The revert timer is anchored to the button: a
    rebuild of the tree deletes it, and an anchored ``singleShot`` is dropped
    rather than firing into a deleted widget.
    """
    button.setText(message)
    QtCore.QTimer.singleShot(
        _COPIED_FLASH_MS, button, lambda: button.setText(revert_to)
    )


def _tree_items(tree: QtWidgets.QTreeWidget):
    """Yield every item in ``tree``, parents before their children."""

    def walk(item):
        yield item
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    for i in range(tree.topLevelItemCount()):
        yield from walk(tree.topLevelItem(i))


def _expanded_keys(tree: QtWidgets.QTreeWidget, key) -> set:
    """Snapshot which branches are open, keyed by ``key(item)``.

    Both archive trees are rebuilt from scratch on every refresh, so the
    ``QTreeWidgetItem`` an expansion belongs to is gone by the time the new one
    exists. Keying by row *identity* rather than position lets the state survive
    that: a batch that gained a cluster, or moved because another one was swept,
    still reopens. Rows whose ``key`` is ``None`` are skipped.
    """
    return {
        k
        for item in _tree_items(tree)
        if item.isExpanded() and (k := key(item)) is not None
    }


def _restore_expanded(tree: QtWidgets.QTreeWidget, keys: set, key) -> None:
    """Reopen the branches named by ``keys`` (the inverse of _expanded_keys)."""
    for item in _tree_items(tree):
        if key(item) in keys:
            item.setExpanded(True)
