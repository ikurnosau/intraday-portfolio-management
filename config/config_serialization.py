import inspect
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import is_dataclass, asdict
from collections.abc import Mapping, Sequence

def serialize_config(obj):
    # 1) primitives & None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # 2) datetimes → ISO
    if isinstance(obj, datetime):
        return obj.isoformat()

    # 3) torch.nn.Module → use its repr (the full architecture dump you love)
    if isinstance(obj, nn.Module):
        return obj

    # 4) torch.optim.Optimizer → grab its defaults + class name
    if isinstance(obj, torch.optim.Optimizer):
        d = dict(obj.defaults)
        d["__class__"] = type(obj).__name__
        return serialize_config(d)

    # 5) torch scheduler → pick the fields you care about, plus class name
    if isinstance(obj, torch.optim.lr_scheduler._LRScheduler):
        sch = {"__class__": type(obj).__name__}
        for attr in ("step_size", "gamma", "T_max", "patience", "factor"):
            if hasattr(obj, attr):
                sch[attr] = getattr(obj, attr)
        return serialize_config(sch)

    # 6) functions / methods → just name
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return {"__class__": "function", "name": obj.__name__}

    # 7) mappings & sequences
    if isinstance(obj, Mapping):
        return {k: serialize_config(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return [serialize_config(v) for v in obj]

    # 8) dataclasses → first turn to dict, then recurse
    if is_dataclass(obj):
        return serialize_config(asdict(obj))

    # 9) any other object with __dict__ → pull public attributes + class
    if hasattr(obj, "__dict__"):
        out = {"__class__": type(obj).__name__}
        for k, v in vars(obj).items():
            if not k.startswith("_"):
                out[k] = serialize_config(v)
        return out

    # 10) fallback to string
    return str(obj)