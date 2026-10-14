"""Content-hash a prompt template (or set of templates) for traceability.

The same pattern had been written independently as PROMPT_HASH in
several agents' prompts.py -- lets a saved result be traced to the
exact prompt version that produced it without a hand-bumped version
number. Written once here instead of copy-pasted again.

Only the fixed template strings should be hashed, never per-call content
(a critique, a user's question, an image) -- the hash identifies which
prompt *scaffolding* produced a result, not the specific call.
"""

import hashlib


def hash_prompts(*templates: str) -> str:
    """Hash one or more prompt template strings in the order given --
    callers must pass them in a stable order across runs (e.g. always
    base template first, then mode variants) or the hash will drift for
    no real reason. Each template is fed into the hasher separately so
    that ("ab", "c") and ("a", "bc") produce different hashes."""
    h = hashlib.sha256()
    for t in templates:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # null separator so ("ab","c") != ("a","bc")
    return h.hexdigest()[:12]
