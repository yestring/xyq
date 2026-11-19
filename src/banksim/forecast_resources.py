"""OS process working-set measurement, including native NumPy/PyTorch allocations."""

import ctypes
import os


def process_memory():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong)] + [(name,ctypes.c_size_t) for name in
                        ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
                         "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        memory = Counters()
        memory.cb = ctypes.sizeof(memory)
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        if not get_info(kernel.GetCurrentProcess(),ctypes.byref(memory),memory.cb):
            raise ctypes.WinError()
        return {"rss_bytes":int(memory.WorkingSetSize),"peak_rss_bytes":int(memory.PeakWorkingSetSize),
                "method":"Windows GetProcessMemoryInfo; process lifetime peak working set, includes native allocations"}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return {"peak_rss_bytes":peak,"method":"resource.ru_maxrss; process lifetime peak"}


def disk_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
