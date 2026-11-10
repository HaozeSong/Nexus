# OCC (eager validation)
Reimplemented from CoAgent (arXiv:2606.15376) §7.1. Read/write sets come from the declared footprints (range reads add the
returned ids). Validation is eager: on each write the middleware checks every in-flight agent; an rw or ww
intersection dooms that agent, which aborts at its next tool call or at commit (rollback = drop its ops,
clear context, restart; cap 5). CoAgent measured this eager variant; Atomix ships a commit-time one
(`protocols/occ_revalidate_retry.py`: revalidate + retry).
