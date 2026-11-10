"""Ablations of Nexus: each one is `nexus.manager.NexusTM` with a few class attributes changed.

Every ablation is selected by its `name`, like any other protocol (`protocols.load_protocol`), so the experiment scripts
and result files refer to them exactly as they refer to the baselines. To add one, subclass NexusTM here, set `name` and
the attributes that differ, and append the class to `ABLATIONS`.
"""
from nexus.manager import NexusTM


# -- repair mechanism ------------------------------------------------------------------------------------------------------
class NexusRestart(NexusTM):
    """An invalidated task restarts from scratch (classic abort/retry)."""
    name = "nexus_restart"
    REPAIR = "restart"


class NexusNotify(NexusTM):
    """Notification-assisted regeneration: the invalidation is appended to the context and the model reconciles
    (MTPO-style repair on the Nexus substrate)."""
    name = "nexus_notify"
    REPAIR = "notify"


class NexusUndo(NexusTM):
    """Selective undo: stale results are refreshed in place, dependent buffered writes retracted, and the model redoes
    them; independent later steps are kept without verification."""
    name = "nexus_undo"
    REPAIR = "undo"


class NexusNoDraft(NexusTM):
    """Regeneration without drafts: the same rollback to the earliest stale read and the same memoization of unchanged
    tool results as anchored replay, but every token of the suffix is decoded again."""
    name = "nexus_nodraft"
    REPAIR = "regen"
    DRAFTS = False


# -- intent policy ---------------------------------------------------------------------------------------------------------
class NexusWait(NexusTM):
    """Readers blocked by an older intent always wait."""
    name = "nexus_wait"
    INTENT_POLICY = "wait"


class NexusProceed(NexusTM):
    """Readers never wait for intents (speculate on the committed value; the read set catches the change at commit)."""
    name = "nexus_proceed"
    INTENT_POLICY = "proceed"


class NexusNoIntents(NexusTM):
    """No write intents at all: no ordering waits and no early invalidation; conflicts are detected only by forward
    validation at commit."""
    name = "nexus_nointents"
    INTENTS = False


# -- engine priority, sampling keys, cancellation --------------------------------------------------------------------------
class NexusNoPriority(NexusTM):
    """No engine priorities for tasks holding intents that others wait on."""
    name = "nexus_nopriority"
    PRIORITIES = False


class NexusPosKeys(NexusTM):
    """Position-keyed instead of content-keyed sampling noise (the engine's deterministic mode)."""
    name = "nexus_poskeys"
    POSITION_KEYS = True


class NexusNoCancel(NexusTM):
    """Forward validation marks the read set stale but does not cancel the in-flight generation (it runs to completion
    and is then rolled back)."""
    name = "nexus_nocancel"
    PREEMPT = False


ABLATIONS = {cls.name: cls for cls in (NexusRestart, NexusNotify, NexusUndo, NexusNoDraft, NexusWait, NexusProceed,
                                       NexusNoIntents, NexusNoPriority, NexusPosKeys, NexusNoCancel)}
