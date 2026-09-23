"""English template families: T4 formal email, T5 chat, T6 key-value, T8 long email (OOD only).

Families only compose fragments from `fields.py`; every slot has sentences for each subset of
its fields. T8 filler sentences contain no digits, so they cannot be mistaken for card values.
"""

from __future__ import annotations

from qf.data.registries import TEMPLATE_FAMILIES
from qf.data.render.base import LayoutFamily

__all__ = ["ChatEn", "FormalEmailEn", "KeyValueEn", "LongEmailEn"]

_NOTES = {
    "weight_conflict": ("{weight_conflict}",),
    "pieces_conflict": ("{pieces_conflict}",),
    "distractors": ("{distractors}",),
}
_EMAIL_SLOTS = {
    "shipper": ("We are writing on behalf of {shipper}.", "This request comes from {shipper}.",
                "Our client {shipper} needs a quote."),
    "route": ("Please quote a shipment {origin_from} {destination_to}.",
              "We need to move cargo {origin_from} {destination_to}.",
              "Please quote a shipment {origin_from}.", "We need to move cargo {destination_to}."),
    "load": ("The load is {pieces} with a total weight of {weight}.", "Total: {pieces}, {weight}.",
             "The load consists of {per_piece}.", "The load is {pieces}.",
             "Total weight is {weight}."),
    "cargo": ("The cargo is {category}.", "Commodity: {category}."),
    "equipment": ("We require a {equipment}.", "Equipment needed: {equipment}."),
    "dates": ("Pickup {pickup}, delivery {delivery}.",
              "We plan pickup {pickup} and delivery {delivery}.",
              "Pickup {pickup}.", "Loading is planned {pickup}.",
              "Delivery is required {delivery}.", "Delivery {delivery}."),
    **_NOTES,
}  # fmt: skip


@TEMPLATE_FAMILIES.register("T4")
class FormalEmailEn(LayoutFamily):
    name = "T4"
    language = "en"
    slots = {
        "greeting": ("Dear colleagues,\n", "Hello,\n", "Good afternoon,\n", "Dear team,\n"),
        "closing": ("Kind regards.", "Thank you in advance.", "Looking forward to your offer."),
        **_EMAIL_SLOTS,
    }
    orders = (
        ("greeting", "shipper", "route", "load", "weight_conflict", "pieces_conflict", "cargo",
         "equipment", "dates", "distractors", "closing"),
        ("greeting", "route", "dates", "load", "weight_conflict", "pieces_conflict", "equipment",
         "cargo", "shipper", "distractors", "closing"),
        ("greeting", "shipper", "cargo", "load", "weight_conflict", "pieces_conflict", "route",
         "equipment", "dates", "distractors", "closing"),
        ("greeting", "route", "equipment", "load", "weight_conflict", "pieces_conflict", "dates",
         "cargo", "distractors", "shipper", "closing"),
        ("greeting", "shipper", "dates", "route", "load", "weight_conflict", "pieces_conflict",
         "equipment", "cargo", "distractors", "closing"),
    )  # fmt: skip


@TEMPLATE_FAMILIES.register("T5")
class ChatEn(LayoutFamily):
    name = "T5"
    language = "en"
    slots = {
        "hi": ("Hi!", "Hey team!", "Hello!"),
        "route": ("Need a truck {origin_from} {destination_to}.",
                  "Got a load {origin_from} {destination_to}.",
                  "Need a truck {origin_from}.", "Got a load going {destination_to}."),
        "load": ("{pieces}, {weight}.", "Weight {weight}, {pieces}.", "It's {per_piece}.",
                 "{pieces}.", "Weight {weight}."),
        "equipment": ("Need a {equipment}.", "Equipment: {equipment}."),
        "dates": ("Pickup {pickup}, drop {delivery}.", "Loading {pickup}, delivery {delivery}.",
                  "Loading {pickup}.", "Pickup {pickup}.", "Drop-off {delivery}."),
        "cargo": ("It's {category}.", "Cargo: {category}."),
        "shipper": ("For {shipper}.", "Client: {shipper}."),
        "end": ("Thanks!", "Rate?", "Let me know."),
        **_NOTES,
    }  # fmt: skip
    orders = (
        ("hi", "route", "load", "weight_conflict", "pieces_conflict", "equipment", "dates",
         "cargo", "shipper", "distractors", "end"),
        ("route", "dates", "load", "weight_conflict", "pieces_conflict", "equipment", "cargo",
         "distractors", "shipper", "end"),
        ("hi", "equipment", "route", "dates", "load", "weight_conflict", "pieces_conflict",
         "shipper", "cargo", "distractors"),
        ("route", "equipment", "load", "weight_conflict", "pieces_conflict", "cargo", "dates",
         "shipper", "distractors", "end"),
        ("hi", "shipper", "route", "cargo", "load", "weight_conflict", "pieces_conflict",
         "equipment", "dates", "distractors", "end"),
    )  # fmt: skip


@TEMPLATE_FAMILIES.register("T6")
class KeyValueEn(LayoutFamily):
    name = "T6"
    language = "en"
    joiner = "\n"
    preposition = False
    slots = {
        "header": ("Transport request", "Shipment details:", "Quote request"),
        "origin": ("Origin: {origin}", "From: {origin}", "Pickup location: {origin}"),
        "destination": ("Destination: {destination}", "To: {destination}",
                        "Delivery location: {destination}"),
        "pickup": ("Pickup date: {pickup}", "Loading date: {pickup}"),
        "delivery": ("Delivery date: {delivery}", "Unloading date: {delivery}"),
        "cargo": ("Commodity: {category}", "Cargo: {category}"),
        "pieces": ("Pieces: {pieces}", "Number of pieces: {pieces}", "Pieces/weight: {per_piece}",
                   "Handling units: {per_piece}"),
        "weight": ("Weight: {weight}", "Gross weight: {weight}"),
        "equipment": ("Equipment: {equipment}", "Trailer type: {equipment}"),
        "shipper": ("Shipper: {shipper}", "Customer: {shipper}"),
        "footer": ("Thanks.", "Please send your rate."),
        **_NOTES,
    }  # fmt: skip
    orders = (
        ("header", "origin", "destination", "pickup", "delivery", "cargo", "pieces", "weight",
         "weight_conflict", "pieces_conflict", "equipment", "shipper", "distractors", "footer"),
        ("header", "shipper", "origin", "destination", "cargo", "weight", "pieces",
         "weight_conflict", "pieces_conflict", "equipment", "pickup", "delivery", "distractors"),
        ("header", "pickup", "origin", "delivery", "destination", "equipment", "cargo", "pieces",
         "weight", "weight_conflict", "pieces_conflict", "shipper", "distractors", "footer"),
        ("origin", "destination", "equipment", "weight", "pieces", "weight_conflict",
         "pieces_conflict", "cargo", "pickup", "delivery", "shipper", "distractors"),
        ("header", "cargo", "pieces", "weight", "weight_conflict", "pieces_conflict", "origin",
         "destination", "pickup", "delivery", "equipment", "distractors", "shipper", "footer"),
    )  # fmt: skip


@TEMPLATE_FAMILIES.register("T8")
class LongEmailEn(LayoutFamily):
    """Long email that retells earlier correspondence around the details; OOD families only."""

    name = "T8"
    language = "en"
    ood_only = True
    slots = {
        "greeting": ("Hi team,\n", "Dear all,\n", "Hello again,\n"),
        "recap": ("Thanks for getting back to us so quickly yesterday.",
                  "Following up on our call earlier this week, here is the new request.",
                  "As discussed with your sales manager, we would like to place another order.",
                  "Sorry for the delay with the details, our warehouse only confirmed them "
                  "this morning."),
        "aside": ("The previous shipment went smoothly, so we would like to work with you again.",
                  "We are still waiting for the documents for the last delivery, but that is a "
                  "separate topic.",
                  "Please note that our office hours have changed."),
        "closing": ("Let me know if you need anything else. Best regards.",
                    "Many thanks in advance, and have a good day.",
                    "Please confirm availability at your earliest convenience."),
        **_EMAIL_SLOTS,
    }  # fmt: skip
    orders = (
        ("greeting", "recap", "shipper", "route", "load", "weight_conflict", "pieces_conflict",
         "aside", "cargo", "equipment", "dates", "distractors", "closing"),
        ("greeting", "recap", "aside", "route", "dates", "load", "weight_conflict",
         "pieces_conflict", "equipment", "cargo", "shipper", "distractors", "closing"),
        ("greeting", "shipper", "recap", "cargo", "load", "weight_conflict", "pieces_conflict",
         "route", "aside", "equipment", "dates", "distractors", "closing"),
        ("greeting", "recap", "route", "equipment", "aside", "load", "weight_conflict",
         "pieces_conflict", "dates", "cargo", "distractors", "shipper", "closing"),
        ("greeting", "aside", "recap", "dates", "route", "load", "weight_conflict",
         "pieces_conflict", "equipment", "cargo", "shipper", "distractors", "closing"),
    )  # fmt: skip
