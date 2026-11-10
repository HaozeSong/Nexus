# Strict 2PL
Reimplemented from CoAgent (arXiv:2606.15376) §7.1 / §7.2. Shared locks on the declared read footprint (range reads lock
`table:*`), exclusive locks on the write footprint, upgrade S→X, locks held until commit. A wait-for graph is
checked on every blocked request; the requester that closes a cycle is the victim (matches the case study in
CoAgent §7.3 where the second requester B is chosen). Victim rollback = drop its ops from the log (saga reverse) and
restart with a cleared context; retry cap 5.
