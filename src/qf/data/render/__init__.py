"""Rendering of LoadFacts into request texts with evidence (step 6).

Importing this package registers the template families (T1-T8) and the hard cases.
"""

from qf.data.render import families_en, families_ru, hard_cases
from qf.data.render.base import (
    HardCaseNotApplicable,
    Layout,
    LayoutFamily,
    RenderError,
    Slot,
    assemble,
    fill_layout,
)
from qf.data.render.fields import Fragment, build_fragments, format_number, render_date
from qf.data.render.hard_cases import DROPPABLE_FIELDS, MIN_WEIGHT_DIFFERENCE_KG
from qf.data.render.vocabulary import ru_plural

__all__ = [
    "DROPPABLE_FIELDS",
    "MIN_WEIGHT_DIFFERENCE_KG",
    "Fragment",
    "HardCaseNotApplicable",
    "Layout",
    "LayoutFamily",
    "RenderError",
    "Slot",
    "assemble",
    "build_fragments",
    "families_en",
    "families_ru",
    "fill_layout",
    "format_number",
    "hard_cases",
    "render_date",
    "ru_plural",
]
