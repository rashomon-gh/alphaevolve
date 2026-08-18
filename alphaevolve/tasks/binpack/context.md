You are designing a priority heuristic for ONLINE 2-resource vector bin
packing. Items arrive one at a time with a (cpu, mem) demand, each in (0, 1].
Every open bin has remaining (cpu, mem) capacity out of (1.0, 1.0). For each
feasible bin (one that fits the item on both resources) your function
`priority(item, remaining)` returns a score; the item is placed into the
feasible bin with the highest score, or a new bin is opened if none fits.

Objective: minimize the number of bins opened, i.e. maximize
utilization = total_demand / (2 * bins_opened). The score dict also reports
the worst trace's utilization, so robust heuristics beat lucky ones.

Known ideas from the bin-packing literature: best fit (choose the tightest
bin), worst fit, dot-product/alignment between the item's demand vector and
the bin's remaining vector (avoids stranding one resource), penalizing bins
that would be left with unusable slivers of a single resource. Combinations
and nonlinear shaping of these signals often beat any single rule.
