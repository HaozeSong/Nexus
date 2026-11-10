# Naive
CoAgent (arXiv:2606.15376) §7.1: "runs them concurrently with no coordination, the 'probably correct' lower bound."
Base `Protocol` with no hooks: every read is served from the live state (materialise(+inf)), every write is
appended to the log in physical order.
