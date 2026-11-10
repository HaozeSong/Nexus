"""Versioned read sets and the staleness tests behind forward validation.

A task's read set is a map  object -> commit timestamp of the state the task saw  (the certificate of the implementation). The
rules that keep it sound when a task reads through its own buffered writes or through listings live in
`VersionedReadSet`; the stale tests used by validation, by the write-premise check and by anchored replay are module functions so
that every protocol configuration applies exactly the same relation.

Objects are `table:id` (one record), `table:*` (a listing / range: it conveys existence, not content) and `table:+` (an insert:
conflicts with listings only; nexus.tools.conflicts).
"""
from __future__ import annotations

from .tools import conflicts, table_level

INVALID = -1          # entry invalidated and not yet re-observed (the task cannot commit or act irreversibly until it re-reads)


def hits(entries, wset) -> set:
    """Read-set entries that a committed write set invalidates. A specific entry is hit by a write of the same object or by a range
    write covering it, never by an insert (which touches no existing record). A listing entry is hit by a range write or an insert
    under it (a record created or deleted), never by an in-place change of an existing record (rule 3, per record)."""
    out = set()
    for x in entries:
        if x.endswith("+"):
            continue
        for w in wset:
            if x.endswith("*"):
                if table_level(w) and conflicts(w, x):
                    out.add(x); break
            elif w == x or (w.endswith("*") and conflicts(w, x)):
                out.add(x); break
    return out


def concrete(stale: set, committed_writes) -> set:
    """The committed objects behind stale entries (what other tasks actually wrote), for naming the records to re-read."""
    return {w for _, w in committed_writes if any(conflicts(w, o) for o in stale)}


def stale(rs: dict, committed_writes) -> set:
    """Entries a later commit has changed, or that were invalidated and not re-observed. A listing entry is stale for a later commit
    under it only while the changed record has not been re-observed individually at or after that commit: a listing plus fresh reads
    of the records that changed since is a consistent view (a record inserted later keeps the listing stale until it is re-listed)."""
    out = {o for o, seen in rs.items() if seen == INVALID}
    for ts, w in committed_writes:
        for o, seen in rs.items():
            if seen == INVALID or ts <= seen or not conflicts(w, o):
                continue
            if o.endswith("*") and w != o and rs.get(w, INVALID) >= ts:
                continue
            out.add(o)
    return out


def stale_for(rs: dict, wset: set, committed_writes) -> set:
    """Stale entries that are a PREMISE of a write on `wset` (rule 4, write-time premise check): an invalidated entry conflicting with
    the write set, or an entry whose covered record a later commit changed where that record conflicts with the write set. A listing
    stale because record r changed does not block a write on another record r' under the same table."""
    out = set()
    for o, seen in rs.items():
        if seen == INVALID:
            if any(conflicts(o, x) for x in wset):
                out.add(o)
            continue
        for ts, w in committed_writes:
            if ts <= seen or not conflicts(w, o):
                continue
            if o.endswith("*") and w != o and rs.get(w, INVALID) >= ts:
                continue
            if any(conflicts(w, x) for x in wset):
                out.add(o); break
    return out


class VersionedReadSet:
    """The read set of one task with the four rules below.

    1. *Version buffered writes*: a write on o creates an entry for o if none exists (grounded in whatever listing covers it, else the
       current timestamp) and *pins* it while the write is buffered, so a later commit on o makes the task's write stale (first
       committer wins also for write-write conflicts).
    2. *Preserve versions when reading own writes*: reading an object the task has buffered a write on returns its own value; the
       entry keeps the version (pin) the write was based on instead of being refreshed to the current timestamp.
    3. *Listings are stale per record*: a listing entry re-observes only covered records that no longer exist; an existing record's
       content is not conveyed by a listing, so its invalidated entry stays until it is re-read.
    4. *Validate write premises* and *retract affected buffered writes* are applied by the manager with `stale_for` and the store.
    """

    def __init__(self):
        self.entries: dict[str, int] = {}
        self.pins: dict[str, int] = {}
        self.explicit: set = set()      # objects observed through READ calls
        self.internal: set = set()      # entries that exist only because of a write's own footprint (go away with the write)

    def reset(self):
        self.__init__()

    # -- observation ----------------------------------------------------------------------------------------------
    def observe(self, objs, now: int, own_writes: set, record_exists):
        """Record an observation of `objs` at commit timestamp `now`. `own_writes` is the task's live buffered write set (rule 2);
        `record_exists(table, id)` is used by rule 3 for deleted records."""
        c = self.entries
        for o in objs:
            own = (not o.endswith("*")) and any(m == o or (m.endswith("*") and conflicts(o, m)) for m in own_writes)
            if own:
                c.setdefault(o, self.pins.get(o, now))
            else:
                c[o] = now
                if o.endswith("*"):
                    for x, seen in list(c.items()):
                        if seen == INVALID and x != o and conflicts(o, x) and not any(m == x for m in own_writes):
                            t, i = x.split(":", 1)
                            if not record_exists(t, i):
                                c[x] = now

    def certify_write(self, wset, now: int):
        """Rule 1: a write claims the object's current version and pins the entry while the write is buffered."""
        c = self.entries
        for o in wset:
            if o.endswith("+"):
                continue
            if o not in c:
                covering = [t for x, t in c.items() if x != o and x.endswith("*") and t != INVALID and conflicts(x, o)]
                c[o] = min(covering) if covering else now
            self.pins.setdefault(o, c[o])

    def invalidate(self, objs):
        for o in list(self.entries):
            if not o.endswith("*") and any(conflicts(x, o) for x in objs):
                self.entries[o] = INVALID

    def unpin(self, live_writes: set):
        """Drop pins of objects the task no longer has live buffered writes on (after a retraction)."""
        for o in list(self.pins):
            if not any(conflicts(o, m) for m in live_writes):
                del self.pins[o]

    def drop_internal(self, live_footprint: set):
        """Entries that only a withdrawn write had created are removed with it: otherwise a stale entry on an object the agent never
        read explicitly could never be refreshed."""
        for o in list(self.internal):
            if o not in live_footprint and o not in self.explicit:
                self.entries.pop(o, None); self.internal.discard(o)
