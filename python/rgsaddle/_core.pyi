"""rgsaddle nanobind surface: status mapping and the ASE image callback."""



def version() -> str:
    """Package version string."""

RGSADDLE_FORCE_L2: int = 0

RGSADDLE_FORCE_LINF: int = 1

RGSADDLE_FORCE_MAX_ATOM: int = 2

RGSADDLE_MINMODE_DIMER: int = 0

RGSADDLE_MINMODE_LANCZOS: int = 1

RGSADDLE_IRC_GS2: int = 0

RGSADDLE_IRC_MOROKUMA: int = 1

class NumericalError(ArithmeticError):
    pass

class ClaimSlot:
    def __init__(self) -> None: ...

    def __enter__(self) -> ClaimSlot:
        """Claim SurfaceAdapter until __exit__."""

    def __exit__(self, exc_type: object | None, exc_value: object | None, traceback: object | None) -> bool:
        """Release the claim. Exceptions propagate."""

def last_error() -> str:
    """
    Thread-local slot reserved for the C ABI last-error text. Returns an empty string unless an error text was set. raise_status reads it and then clears it.
    """

def raise_status(code: int) -> bool:
    """
    Map an rgsaddle_status_t value. The message is the stored text, or the status name when that text is empty. The stored text is cleared. True on OK. False is not used: convergence is not a status code.
    """
