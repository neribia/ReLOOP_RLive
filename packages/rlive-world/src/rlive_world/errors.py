"""Hardware error taxonomy for the world server.

The distinction that matters to a caller is whether retrying the same request
could plausibly succeed:

- A Bolt that is asleep or out of range is missed by one BLE scan and found by the
  next, so the client should retry.
- A camera index that does not exist will never start existing on its own, so
  retrying only burns another Bolt scan and connect before failing identically.

`world_server` maps the two to different status codes (503 vs 422) and the client's
`is_retryable_status` does the rest, so the classification has to be made where the
failure is actually understood -- at the point it is raised, not by pattern-matching
messages in a handler.

Both subclass `RuntimeError` so existing `except RuntimeError` handlers and the
generic 503 fallback keep working for anything not yet classified.
"""


class HardwareError(RuntimeError):
    """Base class for hardware failures surfaced over the API."""


class TransientHardwareError(HardwareError):
    """A failure that may well succeed on a retry.

    Raise this when nothing is wrong with the setup and the operation simply did not
    land this time: a sleeping Bolt that the BLE scan missed, a BLE connect that
    timed out.
    """


class PermanentHardwareError(HardwareError):
    """A failure that will not change without someone intervening.

    Raise this when retrying is pointless: no camera at the requested index, a
    device that is not plugged in. The caller should be told what is wrong rather
    than made to wait through retries.
    """
