"""Nexus (full): anchored replay with drafts, forward validation with engine-side cancellation, write intents with the priced
waiting policy, engine priorities for intent holders."""
from nexus.manager import NexusTM


class P(NexusTM):
    name = "nexus"
