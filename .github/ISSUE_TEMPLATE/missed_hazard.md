---
name: Missed hazard report
about: seedtrace audit failed to flag a real source of nondeterminism
title: "[hazard] "
labels: audit-rule
---

## The hazard

<!-- e.g. a library, op, or language behavior that varies run-to-run
     but the audit stayed silent about it. -->

## Minimal snippet the audit should flag

```python
# this code is nondeterministic but `seedtrace audit` reports nothing
```

## What the fix hint should say

<!-- How would you tell a user to make it deterministic? -->
