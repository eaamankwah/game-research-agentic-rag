"""Persistent long-term memory abstraction for UdaPlay.

Pattern adapted from the supplied M09_demo notebook:
MemoryFragment, MemorySearchResult, TimestampFilter, LongTermMemory,
plus factory functions for binding registration/search tools to an Agent.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

from lib.documents import Document
from lib.tooling import Tool
from lib.vector_db import VectorStoreManager


@dataclass
class MemoryFragment:
    content: str
    owner: str
    namespace: str = "default"
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class MemorySearchResult:
    fragments: List[MemoryFragment]
    metadata: Dict


@dataclass
class TimestampFilter:
    greater_than_value: Optional[int] = None
    lower_than_value: Optional[int] = None


class LongTermMemory:
    """Register and semantically search durable memories."""

    def __init__(self, db: VectorStoreManager,
                 store_name: str = "udaplay_long_term_memory"):
        # get_or_create preserves prior memories instead of deleting them.
        self.vector_store = db.get_or_create_store(store_name)

    def get_namespaces(self) -> List[str]:
        results = self.vector_store.get()
        metadatas = results.get("metadatas", []) or []
        return sorted({
            m.get("namespace", "default") for m in metadatas if m
        })

    def register(self, memory_fragment: MemoryFragment,
                 metadata: Optional[Dict] = None) -> str:
        complete_metadata = {
            "owner": memory_fragment.owner,
            "namespace": memory_fragment.namespace,
            "timestamp": int(memory_fragment.timestamp),
        }
        if metadata:
            complete_metadata.update({
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool))
            })

        self.vector_store.add(Document(
            content=memory_fragment.content,
            metadata=complete_metadata,
        ))
        return "Saved new memory"

    def search(self, query_text: str, owner: str, limit: int = 3,
               timestamp_filter: Optional[TimestampFilter] = None,
               namespace: str = "default") -> MemorySearchResult:
        conditions = [
            {"namespace": {"$eq": namespace}},
            {"owner": {"$eq": owner}},
        ]

        if timestamp_filter:
            if timestamp_filter.greater_than_value is not None:
                conditions.append({
                    "timestamp": {"$gt": int(timestamp_filter.greater_than_value)}
                })
            if timestamp_filter.lower_than_value is not None:
                conditions.append({
                    "timestamp": {"$lt": int(timestamp_filter.lower_than_value)}
                })

        result = self.vector_store.query(
            query_texts=[query_text],
            n_results=limit,
            where={"$and": conditions},
        )

        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        fragments = [
            MemoryFragment(
                content=content,
                owner=meta.get("owner", owner),
                namespace=meta.get("namespace", namespace),
                timestamp=int(meta.get("timestamp", 0)),
            )
            for content, meta in zip(documents, metadatas)
        ]
        return MemorySearchResult(
            fragments=fragments,
            metadata={"distances": distances},
        )


def build_memory_registration_tool(ltm: LongTermMemory,
                                   owner: str,
                                   namespace: str) -> Tool:
    def _register(content: str):
        return ltm.register(MemoryFragment(
            content=content,
            owner=owner,
            namespace=namespace,
        ))

    return Tool(
        func=_register,
        name="register_memory",
        description=(
            "Register durable factual research for use in later sessions.\n"
            "Args:\n"
            "    content: A concise self-contained fact to save"
        ),
    )


def build_memory_search_tool(ltm: LongTermMemory,
                             owner: str,
                             namespace: str) -> Tool:
    def _search(content: str):
        result = ltm.search(
            query_text=content,
            owner=owner,
            namespace=namespace,
            limit=3,
        )
        return [
            {
                "content": fragment.content,
                "owner": fragment.owner,
                "namespace": fragment.namespace,
                "timestamp": fragment.timestamp,
                "distance": distance,
            }
            for fragment, distance in zip(
                result.fragments,
                result.metadata.get("distances", []),
            )
        ]

    return Tool(
        func=_search,
        name="search_memory",
        description=(
            "Search persistent research memory from earlier sessions.\n"
            "Args:\n"
            "    content: The information to look for"
        ),
    )
