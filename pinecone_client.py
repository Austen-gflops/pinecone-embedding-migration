"""
Pinecone Client for both source and target indexes
Optimized for efficient batch fetching
"""
from typing import Dict, List, Optional, Any, Set, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from pinecone import Pinecone
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import config

console = Console()

# Pinecone limits
# Note: While Pinecone docs say 1000, the fetch API uses GET requests
# which have URL length limits. With UUID IDs (~36 chars each),
# 100 IDs is safe; 200 works for most cases.
PINECONE_FETCH_LIMIT = 100  # Safe batch size for fetch (URL length limit)
PINECONE_LIST_PAGE_SIZE = 100  # Default page size for list


@dataclass
class VectorRecord:
    """Represents a vector record from Pinecone"""
    id: str
    values: Optional[List[float]]
    metadata: Dict[str, Any]


class PineconeClient:
    """Client for interacting with Pinecone indexes"""

    def __init__(self):
        self.pc = Pinecone(api_key=config.pinecone.api_key)
        self._source_index = None
        self._target_index = None

    @property
    def source_index(self):
        """Get the source index (gflops-serverless)"""
        if self._source_index is None:
            self._source_index = self.pc.Index(config.pinecone.old_index_name)
            console.print(f"[green]Connected to source index:[/green] {config.pinecone.old_index_name}")
        return self._source_index

    @property
    def target_index(self):
        """Get the target index (askdona)"""
        if self._target_index is None:
            self._target_index = self.pc.Index(config.pinecone.new_index_name)
            console.print(f"[green]Connected to target index:[/green] {config.pinecone.new_index_name}")
        return self._target_index

    def get_namespace_stats(self, index_name: str, namespace: str) -> Dict[str, Any]:
        """Get statistics for a specific namespace in an index"""
        try:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()

            if namespace in stats.namespaces:
                ns_stats = stats.namespaces[namespace]
                return {
                    "namespace": namespace,
                    "vector_count": ns_stats.vector_count,
                    "exists": True
                }
            else:
                return {
                    "namespace": namespace,
                    "vector_count": 0,
                    "exists": False
                }
        except Exception as e:
            console.print(f"[red]Error getting namespace stats:[/red] {e}")
            return {
                "namespace": namespace,
                "vector_count": 0,
                "exists": False,
                "error": str(e)
            }

    def list_all_namespaces(self, index_name: str) -> List[str]:
        """List all namespaces in an index"""
        try:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()
            return list(stats.namespaces.keys())
        except Exception as e:
            console.print(f"[red]Error listing namespaces:[/red] {e}")
            return []

    def get_all_vector_ids(self, namespace: str, source: bool = True) -> Set[str]:
        """Get all vector IDs in a namespace (optimized with pagination)"""
        index = self.source_index if source else self.target_index
        index_name = config.pinecone.old_index_name if source else config.pinecone.new_index_name

        console.print(f"[cyan]Fetching all vector IDs from {index_name} namespace: {namespace}[/cyan]")

        all_ids: Set[str] = set()

        try:
            # Use list to get all vector IDs (automatically paginated)
            for ids_batch in index.list(namespace=namespace):
                all_ids.update(ids_batch)

            console.print(f"[green]Found {len(all_ids)} vector IDs[/green]")
            return all_ids
        except Exception as e:
            console.print(f"[red]Error fetching vector IDs:[/red] {e}")
            return all_ids

    def _fetch_batch(
        self,
        index,
        batch_ids: List[str],
        namespace: str
    ) -> List[VectorRecord]:
        """Fetch a single batch of vectors (for concurrent execution)"""
        records = []
        try:
            response = index.fetch(ids=batch_ids, namespace=namespace)

            for vec_id, vec_data in response.vectors.items():
                records.append(VectorRecord(
                    id=vec_id,
                    values=vec_data.values,
                    metadata=vec_data.metadata or {}
                ))
        except Exception as e:
            console.print(f"[red]Error fetching batch:[/red] {e}")

        return records

    def fetch_vectors_by_ids(
        self,
        ids: List[str],
        namespace: str,
        source: bool = True,
        max_workers: int = 5,
        show_progress: bool = True
    ) -> List[VectorRecord]:
        """
        Fetch vectors by their IDs with optimized concurrent fetching.

        Args:
            ids: List of vector IDs to fetch
            namespace: Namespace to fetch from
            source: If True, fetch from source index; otherwise from target
            max_workers: Number of concurrent fetch threads
            show_progress: Whether to show progress bar

        Returns:
            List of VectorRecord objects
        """
        index = self.source_index if source else self.target_index

        if not ids:
            return []

        # Use maximum batch size (1000) for efficiency
        batch_size = PINECONE_FETCH_LIMIT
        all_records: List[VectorRecord] = []

        # Create batches
        batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        total_batches = len(batches)

        if show_progress:
            console.print(f"[cyan]Fetching {len(ids)} vectors in {total_batches} batches (batch size: {batch_size})...[/cyan]")

        # Use concurrent fetching for better performance
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch fetches
            future_to_batch = {
                executor.submit(self._fetch_batch, index, batch, namespace): i
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                    completed += 1

                    if show_progress:
                        console.print(f"[dim]  Batch {completed}/{total_batches} complete ({len(records)} vectors)[/dim]")

                except Exception as e:
                    console.print(f"[red]Error in batch {batch_idx}:[/red] {e}")

        if show_progress:
            console.print(f"[green]Successfully fetched {len(all_records)} vectors[/green]")

        return all_records

    def fetch_all_vectors_with_metadata(
        self,
        namespace: str,
        source: bool = True,
        progress_callback=None
    ) -> Iterator[VectorRecord]:
        """
        Fetch all vectors with their metadata from a namespace.
        Yields vectors one at a time to handle large datasets.
        """
        # First get all IDs
        all_ids = self.get_all_vector_ids(namespace, source)

        if not all_ids:
            console.print("[yellow]No vectors found in namespace[/yellow]")
            return

        ids_list = list(all_ids)
        total = len(ids_list)
        batch_size = PINECONE_FETCH_LIMIT

        console.print(f"[cyan]Fetching {total} vectors with metadata...[/cyan]")

        for i in range(0, total, batch_size):
            batch_ids = ids_list[i:i + batch_size]
            records = self._fetch_batch(
                self.source_index if source else self.target_index,
                batch_ids,
                namespace
            )

            for record in records:
                yield record

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str
    ) -> int:
        """
        Upsert vectors to the target index.

        Args:
            vectors: List of dicts with 'id', 'values', and 'metadata'
            namespace: Target namespace

        Returns:
            Number of vectors upserted
        """
        if not vectors:
            return 0

        try:
            # Upsert in batches (use config batch size for upsert)
            batch_size = config.migration.upsert_batch_size
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]

                # Format for Pinecone upsert
                pinecone_vectors = [
                    {
                        "id": v["id"],
                        "values": v["values"],
                        "metadata": v["metadata"]
                    }
                    for v in batch
                ]

                self.target_index.upsert(
                    vectors=pinecone_vectors,
                    namespace=namespace
                )

                total_upserted += len(batch)
                time.sleep(config.migration.batch_delay)

            return total_upserted

        except Exception as e:
            console.print(f"[red]Error upserting vectors:[/red] {e}")
            raise

    def check_vector_exists(self, vector_id: str, namespace: str, source: bool = True) -> bool:
        """Check if a vector exists in the index"""
        index = self.source_index if source else self.target_index

        try:
            response = index.fetch(ids=[vector_id], namespace=namespace)
            return vector_id in response.vectors
        except Exception as e:
            console.print(f"[red]Error checking vector:[/red] {e}")
            return False


# Global client instance
pinecone_client = PineconeClient()
