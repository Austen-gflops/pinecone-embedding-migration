"""
Elasticsearch Client for metadata updates

Supports bulk updating of metadata fields in Elasticsearch documents.
NO DELETIONS - only adds new metadata fields.
"""
from typing import Dict, List, Any, Optional, Generator, Set
from dataclasses import dataclass
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, scan
from rich.console import Console
from rich.markup import escape

from config import config

console = Console()

# Elasticsearch batch size for bulk operations
ES_BULK_BATCH_SIZE = 500


@dataclass
class ESDocumentStats:
    """Statistics for Elasticsearch documents"""
    total_count: int
    with_clearance_level: int
    without_clearance_level: int
    namespace: Optional[str] = None


class ElasticsearchClient:
    """Client for interacting with Elasticsearch"""

    def __init__(self):
        self._client: Optional[Elasticsearch] = None

    @property
    def client(self) -> Elasticsearch:
        """Get the Elasticsearch client (lazy initialization)"""
        if self._client is None:
            self._client = Elasticsearch(
                hosts=[config.elasticsearch.endpoint],
                api_key=config.elasticsearch.api_key,
                verify_certs=True,
                ssl_show_warn=False
            )
            console.print(f"[green]Connected to Elasticsearch:[/green] {config.elasticsearch.endpoint}")
        return self._client

    @property
    def index_name(self) -> str:
        """Get the index name from config"""
        return config.elasticsearch.index_name

    def test_connection(self) -> bool:
        """Test the Elasticsearch connection"""
        try:
            info = self.client.info()
            console.print(f"[green]Elasticsearch cluster:[/green] {info['cluster_name']}")
            return True
        except Exception as e:
            console.print(f"[red]Connection failed:[/red] {escape(str(e))}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the index"""
        try:
            # Use count API instead of indices.stats (not available in serverless)
            response = self.client.count(
                index=self.index_name,
                body={"query": {"match_all": {}}}
            )
            doc_count = response['count']
            return {
                "index": self.index_name,
                "doc_count": doc_count,
                "exists": True
            }
        except Exception as e:
            console.print(f"[red]Error getting index stats:[/red] {escape(str(e))}")
            return {
                "index": self.index_name,
                "doc_count": 0,
                "exists": False,
                "error": str(e)
            }

    def list_all_namespaces(self) -> List[str]:
        """Get all unique namespace values from the index"""
        try:
            # Use aggregation to get unique namespace values
            # Try keyword subfield first (for text fields), fallback to direct field
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "namespaces": {
                            "terms": {
                                "field": "pinecone_namespace.keyword",
                                "size": 10000  # Get up to 10k unique namespaces
                            }
                        }
                    }
                }
            )

            namespaces = [
                bucket['key']
                for bucket in response['aggregations']['namespaces']['buckets']
            ]
            return namespaces
        except Exception as e:
            console.print(f"[red]Error listing namespaces:[/red] {escape(str(e))}")
            return []

    def get_namespace_stats(self, namespace: str) -> ESDocumentStats:
        """Get statistics for documents in a specific namespace"""
        try:
            # Count total documents in namespace
            # Use .keyword subfield for exact term matching on text fields
            total_response = self.client.count(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"pinecone_namespace.keyword": namespace}
                    }
                }
            )
            total_count = total_response['count']

            # Count documents WITH clearance_level
            with_clearance_response = self.client.count(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"pinecone_namespace.keyword": namespace}},
                                {"exists": {"field": "clearance_level"}}
                            ]
                        }
                    }
                }
            )
            with_clearance = with_clearance_response['count']

            return ESDocumentStats(
                total_count=total_count,
                with_clearance_level=with_clearance,
                without_clearance_level=total_count - with_clearance,
                namespace=namespace
            )
        except Exception as e:
            console.print(f"[red]Error getting namespace stats:[/red] {escape(str(e))}")
            return ESDocumentStats(
                total_count=0,
                with_clearance_level=0,
                without_clearance_level=0,
                namespace=namespace
            )

    def get_all_stats(self) -> ESDocumentStats:
        """Get statistics for all documents in the index"""
        try:
            # Count total documents
            total_response = self.client.count(
                index=self.index_name,
                body={"query": {"match_all": {}}}
            )
            total_count = total_response['count']

            # Count documents WITH clearance_level
            with_clearance_response = self.client.count(
                index=self.index_name,
                body={
                    "query": {
                        "exists": {"field": "clearance_level"}
                    }
                }
            )
            with_clearance = with_clearance_response['count']

            return ESDocumentStats(
                total_count=total_count,
                with_clearance_level=with_clearance,
                without_clearance_level=total_count - with_clearance,
                namespace=None
            )
        except Exception as e:
            console.print(f"[red]Error getting all stats:[/red] {escape(str(e))}")
            return ESDocumentStats(
                total_count=0,
                with_clearance_level=0,
                without_clearance_level=0,
                namespace=None
            )

    def _generate_update_actions(
        self,
        namespace: Optional[str] = None,
        only_missing: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate update actions for bulk operation.

        Args:
            namespace: Filter by namespace (None = all namespaces)
            only_missing: If True, only update docs without clearance_level

        Yields:
            Update action dictionaries for streaming_bulk
        """
        # Build query
        # Use .keyword subfield for exact term matching on text fields
        if namespace and only_missing:
            query = {
                "bool": {
                    "must": [
                        {"term": {"pinecone_namespace.keyword": namespace}}
                    ],
                    "must_not": [
                        {"exists": {"field": "clearance_level"}}
                    ]
                }
            }
        elif namespace:
            query = {"term": {"pinecone_namespace.keyword": namespace}}
        elif only_missing:
            query = {
                "bool": {
                    "must_not": [
                        {"exists": {"field": "clearance_level"}}
                    ]
                }
            }
        else:
            query = {"match_all": {}}

        # Scan through all matching documents
        for doc in scan(
            client=self.client,
            index=self.index_name,
            query={"query": query},
            _source=False,  # We only need the _id
            scroll='5m',
            size=1000
        ):
            yield {
                "_op_type": "update",
                "_index": self.index_name,
                "_id": doc["_id"],
                "doc": {"clearance_level": config.migration.new_metadata_value}
            }

    def bulk_add_clearance_level(
        self,
        namespace: Optional[str] = None,
        only_missing: bool = True,
        progress_callback=None
    ) -> Dict[str, int]:
        """
        Add clearance_level metadata to documents in bulk.

        Args:
            namespace: Filter by namespace (None = all namespaces)
            only_missing: If True, only update docs without clearance_level
            progress_callback: Optional callback(success, failed, total) for progress

        Returns:
            Dict with 'success' and 'failed' counts
        """
        console.print(f"[cyan]Starting bulk update...[/cyan]")
        if namespace:
            console.print(f"[cyan]Namespace filter:[/cyan] {namespace}")
        if only_missing:
            console.print(f"[cyan]Only updating documents without clearance_level[/cyan]")

        success_count = 0
        failed_count = 0
        total_processed = 0

        try:
            # Use streaming_bulk for memory efficiency
            for ok, result in streaming_bulk(
                client=self.client,
                actions=self._generate_update_actions(namespace, only_missing),
                chunk_size=ES_BULK_BATCH_SIZE,
                raise_on_error=False,
                raise_on_exception=False
            ):
                total_processed += 1
                if ok:
                    success_count += 1
                else:
                    failed_count += 1
                    console.print(f"[red]Failed:[/red] {escape(str(result))}")

                # Progress callback every 100 docs
                if progress_callback and total_processed % 100 == 0:
                    progress_callback(success_count, failed_count, total_processed)

            console.print(f"[green]Bulk update complete![/green]")
            console.print(f"  Success: {success_count}")
            console.print(f"  Failed: {failed_count}")

            return {
                "success": success_count,
                "failed": failed_count
            }

        except Exception as e:
            console.print(f"[red]Bulk update error:[/red] {escape(str(e))}")
            return {
                "success": success_count,
                "failed": failed_count,
                "error": str(e)
            }

    def preview_update(
        self,
        namespace: Optional[str] = None,
        only_missing: bool = True
    ) -> int:
        """
        Preview how many documents would be updated.

        Args:
            namespace: Filter by namespace (None = all namespaces)
            only_missing: If True, only count docs without clearance_level

        Returns:
            Count of documents that would be updated
        """
        try:
            # Build query (same as _generate_update_actions)
            # Use .keyword subfield for exact term matching on text fields
            if namespace and only_missing:
                query = {
                    "bool": {
                        "must": [
                            {"term": {"pinecone_namespace.keyword": namespace}}
                        ],
                        "must_not": [
                            {"exists": {"field": "clearance_level"}}
                        ]
                    }
                }
            elif namespace:
                query = {"term": {"pinecone_namespace.keyword": namespace}}
            elif only_missing:
                query = {
                    "bool": {
                        "must_not": [
                            {"exists": {"field": "clearance_level"}}
                        ]
                    }
                }
            else:
                query = {"match_all": {}}

            response = self.client.count(
                index=self.index_name,
                body={"query": query}
            )
            return response['count']

        except Exception as e:
            console.print(f"[red]Error previewing update:[/red] {escape(str(e))}")
            return 0

    def sample_documents(
        self,
        namespace: Optional[str] = None,
        size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get sample documents for inspection.

        Args:
            namespace: Filter by namespace (None = all namespaces)
            size: Number of documents to return

        Returns:
            List of sample documents
        """
        try:
            # Use .keyword subfield for exact term matching on text fields
            if namespace:
                query = {"term": {"pinecone_namespace.keyword": namespace}}
            else:
                query = {"match_all": {}}

            response = self.client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": size,
                    "_source": ["pinecone_namespace", "clearance_level", "filename", "title", "vector_id"]
                }
            )

            return [hit["_source"] for hit in response["hits"]["hits"]]

        except Exception as e:
            console.print(f"[red]Error getting sample documents:[/red] {escape(str(e))}")
            return []


# Global client instance
elasticsearch_client = ElasticsearchClient()
