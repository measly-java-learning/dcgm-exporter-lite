"""Background watcher for critical XID events."""

from __future__ import annotations

import logging
import threading
import time

from .nvml import XidEventSource

logger = logging.getLogger(__name__)

_WAIT_MS = 500
_ERROR_BACKOFF_S = 1.0
_ERROR_LOG_INTERVAL_S = 300.0

# From DCGM Exporter 4.9.0 internal/pkg/collector/xid_errors.go.
XID_MESSAGES: dict[int, str] = {
    0: "No Error",
    1: "Invalid or corrupted push buffer stream",
    2: "Invalid or corrupted push buffer stream",
    3: "Invalid or corrupted push buffer stream",
    4: "Invalid or corrupted push buffer stream",
    5: "Unused",
    6: "Invalid or corrupted push buffer stream",
    7: "Invalid or corrupted push buffer address",
    8: "GPU stopped processing",
    9: "Driver error programming GPU",
    10: "Unused",
    11: "Invalid or corrupted push buffer stream",
    12: "Driver error handling GPU exception",
    13: "Graphics Engine Exception",
    14: "Unused",
    15: "Unused",
    16: "Display engine hung",
    17: "Unused",
    18: "Bus mastering disabled in PCI Config Space",
    19: "Display Engine error",
    20: "Invalid or corrupted Mpeg push buffer",
    21: "Invalid or corrupted Motion Estimation push buffer",
    22: "Invalid or corrupted Video Processor push buffer",
    23: "Unused",
    24: "GPU semaphore timeout",
    25: "Invalid or illegal push buffer stream",
    26: "Framebuffer timeout",
    27: "Video processor exception",
    28: "Video processor exception",
    29: "Video processor exception",
    30: "GPU semaphore access error",
    31: "GPU memory page fault",
    32: "Invalid or corrupted push buffer stream",
    33: "Internal micro-controller error",
    34: "Video processor exception",
    35: "Video processor exception",
    36: "Video processor exception",
    37: "Driver firmware error",
    38: "Driver firmware error",
    39: "Unused",
    40: "Unused",
    41: "Unused",
    42: "Video processor exception",
    43: "GPU stopped processing",
    44: "Graphics Engine fault during context switch",
    45: (
        "Preemptive cleanup, due to previous errors -- Most likely to see when "
        "running multiple cuda"
    ),
    46: "GPU stopped processing",
    47: "Video processor exception",
    48: "Double Bit ECC Error",
    49: "Unused",
    50: "Unused",
    51: "Unused",
    52: "Unused",
    53: "Unused",
    54: "Auxiliary power is not connected to the GPU board",
    55: "Unused",
    56: "Display Engine error",
    57: "Error programming video memory interface",
    58: "Unstable video memory interface detected",
    59: "Internal micro-controller error",
    60: "Video processor exception",
    61: "Internal micro-controller breakpoint/warning",
    62: "Internal micro-controller halt",
    63: "ECC page retirement or row remapping recording event",
    64: "ECC page retirement or row remapper recording failure",
    65: "Video processor exception",
    66: "Illegal access by driver",
    67: "Illegal access by driver",
    68: "NVDEC0 Exception",
    69: "Graphics Engine class error",
    70: "CE3: Unknown Error",
    71: "CE4: Unknown Error",
    72: "CE5: Unknown Error",
    73: "NVENC2 Error",
    74: "NVLINK Error",
    75: "CE6: Unknown Error",
    76: "CE7: Unknown Error",
    77: "CE8: Unknown Error",
    78: "vGPU Start Error",
    79: "GPU has fallen off the bus",
    80: "Corrupted data sent to GPU",
    81: "VGA Subsystem Error",
    82: "NVJPG0 Error",
    83: "NVDEC1 Error",
    84: "NVDEC2 Error",
    85: "CE9: Unknown Error",
    86: "OFA Exception",
    87: "Reserved",
    88: "NVDEC3 Error",
    89: "NVDEC4 Error",
    90: "Reserved",
    91: "Reserved",
    92: "High single-bit ECC error rate",
    93: "Non-fatal violation of provisioned InfoROM wear limit",
    94: "Contained ECC error",
    95: "Uncontained ECC error",
    96: "NVDEC5 Error",
    97: "NVDEC6 Error",
    98: "NVDEC7 Error",
    99: "NVJPG1 Error",
    100: "NVJPG2 Error",
    101: "NVJPG3 Error",
    102: "NVJPG4 Error",
    103: "NVJPG5 Error",
    104: "NVJPG6 Error",
    105: "NVJPG7 Error",
    106: "SMBPBI Test Message",
    107: "SMBPBI Test Message Silent",
    108: "Reserved",
    109: "Context Switch Timeout Error",
    110: "Security Fault Error",
    111: "Display Bundle Error Event",
    112: "Display Supervisor Error",
    113: "DP Link Training Error",
    114: "Display Pipeline Underflow Error",
    115: "Display Core Channel Error",
    116: "Display Window Channel Error",
    117: "Display Cursor Channel Error",
    118: "Display Pixel Pipeline Error",
    119: "GSP RPC Timeout",
    120: "GSP Error",
    121: "C2C Link Error",
    122: "SPI PMU RPC Read Failure",
    123: "SPI PMU RPC Write Failure",
    124: "SPI PMU RPC Erase Failure",
    125: "Inforom FS Failure",
    126: "Robust Channel CE10 Error",
    127: "Robust Channel CE11 Error",
    128: "Robust Channel CE12 Error",
    129: "Robust Channel CE13 Error",
    130: "Robust Channel CE14 Error",
    131: "Robust Channel CE15 Error",
    132: "Robust Channel CE16 Error",
    133: "Robust Channel CE17 Error",
    134: "Robust Channel CE18 Error",
    135: "Robust Channel CE19 Error",
    136: "ALI Training Fail",
    137: "NVLink Privilege Error",
    138: "Robust Channel DLA Error",
    139: "Robust Channel OFA1 Error",
    140: "Unrecovered ECC Error",
    141: "Robust Channel Fast Path Error",
    142: "Robust Channel NVENVC3 Error",
    143: "GPU Initialization Failure",
    144: "NVLINK: Saw Error",
    145: "NVLINK: RLW Error",
    146: "NVLINK: TLW Error",
    147: "NVLINK: TREX Error",
    148: "NVLINK: NVLPW_CTRL Error",
    149: "NVLINK: NETIR Error",
    150: "NVLINK: MSE Error",
    151: "Robust Channel Key rotation Error",
    152: "Robust Channel DLA SMMU Error",
    153: "Robust Channel DLA timeout Error",
    154: "GPU Recovery Action Changed",
    155: "NVLINK: SW Defined Error",
    156: "Resource Retirement Event",
    157: "Resource Retirement Failure",
    158: "GPU Fatal Timeout",
    159: "Robust Channel CHI Non-Data Error",
    160: "Channel Retirement Event",
    161: "Channel Retirement Failure",
    162: "Power Smoothing HW Circuitry Reengaged",
    163: "Power Smoothing HW Circuitry Disengaged",
    164: "Power Smoothing HW Circuitry Low Lifetime",
    165: "Power Smoothing HW Circuitry Lifetime Exhausted",
    166: "NVLINK Secure Crypto Error",
    167: "PCIE Fatal Timeout",
    168: "Reduced GPU Memory Capacity (Errors found in write protection region)",
    169: "Internal micro-controller halt",
    170: "NVLINK: Interrupt seen in CC mode",
    171: "Uncorrectable DRAM Error",
    172: "Uncorrectable SRAM Error",
}


def xid_message(code: int) -> str:
    return XID_MESSAGES.get(code, "Unknown Error")


class XidWatcher(threading.Thread):
    """Record the last XID code per GPU from an NVML event set."""

    def __init__(
        self, source: XidEventSource, *, error_backoff_s: float = _ERROR_BACKOFF_S
    ) -> None:
        super().__init__(name="dcgm-xid-watcher", daemon=True)
        self._source = source
        self._error_backoff_s = error_backoff_s
        self._last_error_log: float | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last: dict[int, int] = {}
        self.available = bool(getattr(source, "available", True))
        registered = getattr(source, "registered", None)
        self._registered = None if registered is None else frozenset(registered)

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._source.wait(_WAIT_MS)
            except Exception as exc:
                now = time.monotonic()
                if (
                    self._last_error_log is None
                    or now - self._last_error_log >= _ERROR_LOG_INTERVAL_S
                ):
                    self._last_error_log = now
                    logger.warning("XID event source failed: %s", exc)
                # Back off so a persistently failing source (e.g. a GPU that
                # fell off the bus) cannot spin this thread.
                self._stop_event.wait(self._error_backoff_s)
                continue

            if event is None:
                continue

            index, code = event
            with self._lock:
                self._last[index] = code

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self.join(timeout)
        self._source.close()

    def watches(self, index: int) -> bool:
        """True when XID events for GPU ``index`` are registered."""
        if not self.available:
            return False
        return self._registered is None or index in self._registered

    def last(self, index: int) -> int | None:
        with self._lock:
            return self._last.get(index)
