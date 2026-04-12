"""
Build Type Registry (ARCH-18).

Maps build type names to their (assembly_graph, retrieval_graph) pairs.
Graphs are compiled lazily on first use and cached thereafter.

CR-A04: Graph compilation uses asyncio.to_thread() to avoid blocking the
event loop with the synchronous threading.Lock + builder() call.

Usage:
    from app.flows.build_type_registry import get_assembly_graph, get_retrieval_graph

    graph = await get_assembly_graph("tiered-summary")
    result = await graph.ainvoke(input_state)
"""

import asyncio
import logging
import threading
from typing import Any, Callable

_log = logging.getLogger("emad_host.flows.build_type_registry")

# R5-m2: Lock for thread-safe registration and lazy compilation.
_lock = threading.Lock()

# Registry: name -> (assembly_builder, retrieval_builder)
_registry: dict[str, tuple[Callable, Callable]] = {}

# Compiled graph cache: (name, "assembly"|"retrieval") -> compiled graph
_compiled_cache: dict[tuple[str, str], Any] = {}


def register_build_type(
    name: str,
    assembly_builder: Callable,
    retrieval_builder: Callable,
) -> None:
    """Register a build type with its assembly and retrieval graph builders.

    Args:
        name: Build type name (e.g., "sliding-window", "tiered-summary").
        assembly_builder: Callable that returns a compiled StateGraph for assembly.
        retrieval_builder: Callable that returns a compiled StateGraph for retrieval.
    """
    with _lock:
        if name in _registry:
            _log.warning("Build type '%s' already registered — overwriting", name)
        _registry[name] = (assembly_builder, retrieval_builder)
    _log.info("Registered build type: %s", name)


def _get_graph_sync(name: str, kind: str) -> Any:
    """Compile and cache a graph under the lock (sync, for use in executor).

    Args:
        name: Build type name.
        kind: "assembly" or "retrieval".
    """
    idx = 0 if kind == "assembly" else 1
    cache_key = (name, kind)
    with _lock:
        if cache_key in _compiled_cache:
            return _compiled_cache[cache_key]

        if name not in _registry:
            raise ValueError(
                f"Build type '{name}' is not registered. "
                f"Available: {list(_registry.keys())}"
            )

        builder = _registry[name][idx]
        graph = builder()
        _compiled_cache[cache_key] = graph
    _log.info("Compiled %s graph for build type: %s", kind, name)
    return graph


async def get_assembly_graph(name: str) -> Any:
    """Return the compiled assembly graph for a build type (lazy, async).

    CR-A04: Compilation is offloaded to a thread to avoid blocking the
    event loop. Cache hits return immediately without thread dispatch.

    Raises ValueError if the build type is not registered.
    """
    cache_key = (name, "assembly")
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]
    return await asyncio.to_thread(_get_graph_sync, name, "assembly")


async def get_retrieval_graph(name: str) -> Any:
    """Return the compiled retrieval graph for a build type (lazy, async).

    CR-A04: Compilation is offloaded to a thread to avoid blocking the
    event loop. Cache hits return immediately without thread dispatch.

    Raises ValueError if the build type is not registered.
    """
    cache_key = (name, "retrieval")
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]
    return await asyncio.to_thread(_get_graph_sync, name, "retrieval")


def list_build_types() -> list[str]:
    """Return a list of all registered build type names."""
    return list(_registry.keys())


def clear_compiled_cache() -> None:
    """Clear compiled graph caches only (not the registry).

    Called after install_stategraph() to ensure next invocation
    recompiles graphs from updated packages. The registry itself
    is repopulated by stategraph_registry.scan().
    """
    with _lock:
        _compiled_cache.clear()
    _log.info("Compiled graph caches cleared")
