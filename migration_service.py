"""
Migration Service for transferring embeddings from old index to new index
"""
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import json
import time
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from config import config
from pinecone_client import pinecone_client, VectorRecord
from gemini_client import gemini_client

console = Console()


@dataclass
class MigrationStats:
    """Statistics for migration progress"""
    total_source_vectors: int = 0
    already_migrated: int = 0
    to_migrate: int = 0
    successfully_migrated: int = 0
    failed: int = 0
    skipped_no_text: int = 0


class MigrationService:
    """Service for migrating vector embeddings between Pinecone indexes"""

    def __init__(self):
        self.pinecone = pinecone_client
        self.gemini = gemini_client
        self.stats = MigrationStats()

    def display_migration_info(self):
        """Display migration configuration information"""
        console.print(config.display_info())

        # Display the new metadata that will be added
        panel = Panel(
            f"""[bold cyan]New Metadata Added to All Embeddings:[/bold cyan]

    [yellow]Key:[/yellow]   {config.migration.new_metadata_key}
    [yellow]Value:[/yellow] {config.migration.new_metadata_value}

This metadata will be added to EVERY embedding during migration.
All original metadata will be preserved.""",
            title="Migration Metadata",
            border_style="green"
        )
        console.print(panel)

    def get_namespace_count(self, namespace: str, index_name: str) -> int:
        """Get the count of vectors in a namespace"""
        stats = self.pinecone.get_namespace_stats(index_name, namespace)
        return stats.get("vector_count", 0)

    def display_namespace_comparison(self, namespace: str):
        """Display comparison of namespace between source and target"""
        source_stats = self.pinecone.get_namespace_stats(
            config.pinecone.old_index_name, namespace
        )
        target_stats = self.pinecone.get_namespace_stats(
            config.pinecone.new_index_name, namespace
        )

        table = Table(title=f"Namespace Comparison: {namespace}")
        table.add_column("Property", style="cyan")
        table.add_column(f"Source ({config.pinecone.old_index_name})", style="yellow")
        table.add_column(f"Target ({config.pinecone.new_index_name})", style="green")

        table.add_row(
            "Vector Count",
            str(source_stats.get("vector_count", 0)),
            str(target_stats.get("vector_count", 0))
        )
        table.add_row(
            "Exists",
            "[green]Yes[/green]" if source_stats.get("exists") else "[red]No[/red]",
            "[green]Yes[/green]" if target_stats.get("exists") else "[red]No[/red]"
        )

        console.print(table)

        # Show remaining to migrate
        remaining = source_stats.get("vector_count", 0) - target_stats.get("vector_count", 0)
        if remaining > 0:
            console.print(f"\n[yellow]Estimated vectors to migrate: {remaining}[/yellow]")
        elif remaining == 0 and source_stats.get("vector_count", 0) > 0:
            console.print("\n[green]All vectors appear to be migrated![/green]")

    def get_vectors_to_migrate(self, namespace: str) -> Tuple[List[VectorRecord], Set[str]]:
        """
        Get vectors that need to be migrated (not already in target).

        Returns:
            Tuple of (vectors_to_migrate, already_migrated_ids)
        """
        console.print("\n[bold cyan]Step 1: Fetching existing IDs from target index...[/bold cyan]")

        # Get IDs already in target
        target_ids = self.pinecone.get_all_vector_ids(namespace, source=False)
        self.stats.already_migrated = len(target_ids)
        console.print(f"[green]Found {len(target_ids)} vectors already in target[/green]")

        console.print("\n[bold cyan]Step 2: Fetching source vectors...[/bold cyan]")

        # Get all source IDs
        source_ids = self.pinecone.get_all_vector_ids(namespace, source=True)
        self.stats.total_source_vectors = len(source_ids)
        console.print(f"[green]Found {len(source_ids)} vectors in source[/green]")

        # Find IDs to migrate (in source but not in target)
        ids_to_migrate = source_ids - target_ids
        self.stats.to_migrate = len(ids_to_migrate)

        console.print(f"[yellow]Vectors to migrate: {len(ids_to_migrate)}[/yellow]")

        if not ids_to_migrate:
            console.print("[green]No new vectors to migrate![/green]")
            return [], target_ids

        console.print("\n[bold cyan]Step 3: Fetching vector metadata from source...[/bold cyan]")

        # Fetch the vectors that need migration
        vectors_to_migrate = self.pinecone.fetch_vectors_by_ids(
            list(ids_to_migrate), namespace, source=True
        )

        return vectors_to_migrate, target_ids

    def prepare_vector_for_upsert(
        self,
        original: VectorRecord,
        new_embedding: List[float]
    ) -> Dict[str, Any]:
        """
        Prepare a vector for upserting to target index.

        Preserves original ID and metadata, adds new metadata,
        and uses new Gemini embedding.
        """
        # Copy original metadata
        new_metadata = dict(original.metadata)

        # Add new clearance_level metadata
        new_metadata[config.migration.new_metadata_key] = config.migration.new_metadata_value

        return {
            "id": original.id,
            "values": new_embedding,
            "metadata": new_metadata
        }

    def migrate_namespace(
        self,
        namespace: str,
        dry_run: bool = False,
        batch_size: int = 50
    ) -> MigrationStats:
        """
        Migrate all vectors in a namespace from source to target.

        Args:
            namespace: The namespace to migrate
            dry_run: If True, don't actually upsert, just show what would be done
            batch_size: Number of vectors to process at a time

        Returns:
            Migration statistics
        """
        self.stats = MigrationStats()

        console.print(Panel(
            f"[bold]Starting migration for namespace:[/bold] {namespace}",
            title="Migration Started",
            border_style="blue"
        ))

        start_time = datetime.now()

        # Get vectors to migrate
        vectors_to_migrate, already_migrated = self.get_vectors_to_migrate(namespace)

        if not vectors_to_migrate:
            console.print("\n[green]Migration complete - no new vectors to process[/green]")
            return self.stats

        if dry_run:
            console.print("\n[yellow]DRY RUN MODE - No changes will be made[/yellow]")

        console.print(f"\n[bold cyan]Step 4: Processing {len(vectors_to_migrate)} vectors...[/bold cyan]")

        # Process in batches
        total_batches = (len(vectors_to_migrate) + batch_size - 1) // batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "Migrating vectors...",
                total=len(vectors_to_migrate)
            )

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(vectors_to_migrate))
                batch = vectors_to_migrate[start_idx:end_idx]

                # Extract texts for embedding
                texts = []
                valid_vectors = []
                for vec in batch:
                    text = vec.metadata.get("text", "")
                    if text and text.strip():
                        texts.append(text)
                        valid_vectors.append(vec)
                    else:
                        self.stats.skipped_no_text += 1
                        console.print(f"[yellow]Skipping vector {vec.id} - no text content[/yellow]")

                if not texts:
                    progress.update(task, advance=len(batch))
                    continue

                # Generate new embeddings
                try:
                    embeddings = self.gemini.generate_batch_embeddings(texts, show_progress=False)
                except Exception as e:
                    console.print(f"[red]Error generating embeddings for batch {batch_idx + 1}:[/red] {e}")
                    self.stats.failed += len(valid_vectors)
                    progress.update(task, advance=len(batch))
                    continue

                # Prepare vectors for upsert
                vectors_to_upsert = []
                for vec, embedding in zip(valid_vectors, embeddings):
                    if embedding is not None:
                        prepared = self.prepare_vector_for_upsert(vec, embedding)
                        vectors_to_upsert.append(prepared)
                    else:
                        self.stats.failed += 1
                        console.print(f"[yellow]Failed to generate embedding for vector {vec.id}[/yellow]")

                # Upsert to target
                if vectors_to_upsert and not dry_run:
                    try:
                        upserted = self.pinecone.upsert_vectors(vectors_to_upsert, namespace)
                        self.stats.successfully_migrated += upserted
                    except Exception as e:
                        console.print(f"[red]Error upserting batch {batch_idx + 1}:[/red] {e}")
                        self.stats.failed += len(vectors_to_upsert)
                elif vectors_to_upsert and dry_run:
                    self.stats.successfully_migrated += len(vectors_to_upsert)
                    console.print(f"[dim]Would upsert {len(vectors_to_upsert)} vectors[/dim]")

                progress.update(task, advance=len(batch))

                # Small delay between batches
                if batch_idx < total_batches - 1:
                    time.sleep(config.migration.batch_delay)

        # Calculate duration
        duration = datetime.now() - start_time

        # Display final stats
        self._display_final_stats(namespace, duration, dry_run)

        return self.stats

    def _display_final_stats(self, namespace: str, duration, dry_run: bool):
        """Display final migration statistics"""
        table = Table(title=f"Migration Results for {namespace}")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="yellow")

        table.add_row("Total Source Vectors", str(self.stats.total_source_vectors))
        table.add_row("Already Migrated (Skipped)", str(self.stats.already_migrated))
        table.add_row("Needed Migration", str(self.stats.to_migrate))
        table.add_row(
            "Successfully Migrated" + (" (Dry Run)" if dry_run else ""),
            f"[green]{self.stats.successfully_migrated}[/green]"
        )
        table.add_row("Failed", f"[red]{self.stats.failed}[/red]")
        table.add_row("Skipped (No Text)", f"[yellow]{self.stats.skipped_no_text}[/yellow]")
        table.add_row("Duration", str(duration))

        console.print("\n")
        console.print(table)

        # Final verification
        if not dry_run:
            console.print("\n[bold cyan]Verifying migration...[/bold cyan]")
            self.display_namespace_comparison(namespace)


# Global service instance
migration_service = MigrationService()
