from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple, Type, Literal

from datatype import TensorShape

@dataclass
class _CachedTensor:
    name: str
    cached_size: int
    dirty: bool = False

class InterOpCacheSimulator:
    "notice: all size unit: var (e.g. fp16: 1 var = 2 byte)"
    def __init__(self, capacity: int, mode: Literal['write-back', 'write-through'] = 'write-back', count_write_back=True) -> None:
        self.cached: OrderedDict[str, _CachedTensor] = OrderedDict()
        self.used = 0
        self.capacity = int(capacity)
        self.mode = mode
        self.count_write_back = count_write_back

    def _evict(self, size) -> int:
        """return write-back size (always 0 if not in 'write-back' mode)"""
        assert size <= self.used
        self.used -= size
        write_back_size = 0
        for t in list(self.cached.values()):
            if size == 0:
                break
            evictd_size = min(t.cached_size, size)
            if t.dirty:
                write_back_size += evictd_size
            t.cached_size -= evictd_size
            size -= evictd_size
            if t.cached_size == 0:
                self.cached.popitem(last=False)
        return write_back_size

    def _cache(self, tensor_name: str, tensor_size: int) -> Tuple[int, int, int]:
        "return (already_cached, evicted, write_back)"
        if not self.capacity:
            return 0, 0, 0

        size_to_cache = min(tensor_size, self.capacity)
        if tensor_name in self.cached:
            cached_size = self.cached.pop(tensor_name).cached_size
            self.used -= cached_size
        else:
            cached_size = 0

        size_to_evict = max(size_to_cache - (self.capacity - self.used), 0)
        # print(f"size_to_cache: {size_to_cache/1e6:.3f}, free cache: {(self.capacity - self.used)/1e6:.3f}, size_to_evict: {size_to_evict/1e6:.3f}")
        write_back = self._evict(size_to_evict)
        self.cached[tensor_name] = _CachedTensor(tensor_name, size_to_cache)
        self.used += size_to_cache

        return cached_size, size_to_evict, write_back

    def read(self, tensor_name: str, tensor_size: int) -> Tuple[int, int, int, int]:
        "return (memory, cached, evicted), 'memory' is required memory access size of the tensor (0 if whole tensor is cached), include write-back size (if any, according to config)"
        if not self.capacity:
            return tensor_size, 0, 0, 0

        cached, evicted, write_back = self._cache(tensor_name, tensor_size)
        if self.mode == 'write-back' and self.count_write_back:
            memory = tensor_size - cached + write_back
        else:
            memory = tensor_size - cached
        return memory, cached, evicted, write_back

    def write(self, tensor_name: str, tensor_size: int) -> Tuple[int, int, int]:
        "return (memory, evicted), 'memory' is required memory access size of the tensor (0 if whole tensor is cached), include write-back size (if any, according to config)"
        if not self.capacity:
            return tensor_size, 0, 0

        size_to_cache = min(tensor_size, self.capacity)
        _, evicted, write_back = self._cache(tensor_name, tensor_size)
        if self.mode == 'write-back':
            self.cached[tensor_name].dirty = True
            if self.count_write_back:
                memory = tensor_size - size_to_cache + write_back
            else:
                memory = tensor_size - size_to_cache
        else:
            memory = tensor_size
        return memory, evicted, write_back
