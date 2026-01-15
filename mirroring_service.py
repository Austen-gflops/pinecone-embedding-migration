"""
Mirroring Service for continuous synchronization between Pinecone indexes

This service extends migration functionality to support:
1. Migrating missing records (source → target)
2. Detecting and updating records with metadata differences

IMPORTANT: Source index (gflops-serverless) is ALWAYS read-only.
Only the target index (askdona) is ever modified.
"""
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
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


# Keys to ignore when comparing metadata between source and target
# These are added during migration and don't exist in the source
METADATA_IGNORE_KEYS = {'clearance_level'}


@dataclass
class MirroringStats:
    """Statistics for namespace mirroring progress"""
    # Source statistics
    total_source_vectors: int = 0

    # Target statistics
    total_target_vectors: int = 0

    # Analysis results
    missing_in_target: int = 0      # Need to migrate (new vectors)
    matching_ids: int = 0           # Exist in both indexes
    metadata_matches: int = 0       # Matching IDs with identical metadata
    metadata_differs: int = 0       # Matching IDs with different metadata

    # Processing results
    successfully_migrated: int = 0  # New vectors added to target
    successfully_updated: int = 0   # Existing vectors updated in target
    failed_migration: int = 0       # Failed to add new vectors
    failed_update: int = 0          # Failed to update existing vectors
    skipped_no_text: int = 0        # Skipped due to missing text content

    # Detailed tracking for metadata differences
    metadata_diff_details: List[Dict[str, Any]] = field(default_factory=list)


def compare_metadata(
    source_metadata: Dict[str, Any],
    target_metadata: Dict[str, Any],
    ignore_keys: Set[str] = None
) -> Tuple[bool, List[str]]:
    """
    Compare metadata between source and target vectors.

    Args:
        source_metadata: Metadata from source vector
        target_metadata: Metadata from target vector
        ignore_keys: Keys to ignore in comparison (default: clearance_level)

    Returns:
        Tuple of (differs: bool, diff_keys: List[str])
        - differs: True if there are meaningful differences
        - diff_keys: List of keys that differ
    """
    if ignore_keys is None:
        ignore_keys = METADATA_IGNORE_KEYS

    diff_keys = []

    # Get comparable keys from both sides
    source_keys = set(source_metadata.keys()) - ignore_keys
    target_keys = set(target_metadata.keys()) - ignore_keys

    # Check for keys only in source (new keys added to source)
    keys_only_in_source = source_keys - target_keys
    if keys_only_in_source:
        diff_keys.extend([f"+{k}" for k in keys_only_in_source])

    # Check for keys only in target (keys removed from source)
    keys_only_in_target = target_keys - source_keys
    if keys_only_in_target:
        diff_keys.extend([f"-{k}" for k in keys_only_in_target])

    # Compare values for common keys
    common_keys = source_keys & target_keys
    for key in common_keys:
        source_val = source_metadata.get(key)
        target_val = target_metadata.get(key)

        # Handle different types gracefully
        if source_val != target_val:
            diff_keys.append(f"~{key}")

    return len(diff_keys) > 0, diff_keys


class MirroringService:
    """
    Service for mirroring vector embeddings between Pinecone indexes.

    Supports two operations:
    1. Migration: Add new vectors that exist in source but not target
    2. Update: Refresh vectors whose metadata has changed in source

    IMPORTANT: Source index is NEVER modified. All changes go to target only.
    """

    def __init__(self):
        self.pinecone = pinecone_client
        self.gemini = gemini_client
        self.stats = MirroringStats()

    def display_mirroring_info(self):
        """Display mirroring configuration information"""
        panel = Panel(
            f"""[bold cyan]Namespace Mirroring Configuration[/bold cyan]

[yellow]Source Index (READ-ONLY):[/yellow]
  Name: {config.pinecone.old_index_name}

[yellow]Target Index (WRITABLE):[/yellow]
  Name: {config.pinecone.new_index_name}

[yellow]Mirroring Operations:[/yellow]
  1. Migrate missing vectors (source → target)
  2. Update vectors with metadata differences

[yellow]Metadata Added During Sync:[/yellow]
  Key: {config.migration.new_metadata_key}
  Value: {config.migration.new_metadata_value}

[yellow]Ignored in Comparison:[/yellow]
  {', '.join(METADATA_IGNORE_KEYS)}

[dim]The source index is NEVER modified.
Only the target index receives changes.[/dim]""",
            title="Mirroring Configuration",
            border_style="blue"
        )
        console.print(panel)

    def analyze_namespace(
        self,
        namespace: str,
        sample_size: int = 5,
        show_progress: bool = True
    ) -> MirroringStats:
        """
        Analyze a namespace to determine mirroring needs.

        This is a read-only analysis that:
        1. Identifies vectors missing in target
        2. Compares metadata for matching vectors
        3. Reports differences without making changes

        Args:
            namespace: Namespace to analyze
            sample_size: Number of metadata differences to sample for details
            show_progress: Whether to show progress indicators

        Returns:
            MirroringStats with analysis results
        """
        self.stats = MirroringStats()

        console.print(Panel(
            f"[bold]Analyzing namespace:[/bold] {namespace}",
            title="Namespace Analysis",
            border_style="cyan"
        ))

        # Step 1: Get all IDs from source
        console.print("\n[bold cyan]Step 1/4: Fetching source vector IDs...[/bold cyan]")
        source_ids = self.pinecone.get_all_vector_ids(namespace, source=True)
        self.stats.total_source_vectors = len(source_ids)
        console.print(f"[green]Found {len(source_ids)} vectors in source[/green]")

        # Step 2: Get all IDs from target
        console.print("\n[bold cyan]Step 2/4: Fetching target vector IDs...[/bold cyan]")
        target_ids = self.pinecone.get_all_vector_ids(namespace, source=False)
        self.stats.total_target_vectors = len(target_ids)
        console.print(f"[green]Found {len(target_ids)} vectors in target[/green]")

        # Step 3: Categorize IDs
        console.print("\n[bold cyan]Step 3/4: Categorizing vectors...[/bold cyan]")

        missing_in_target = source_ids - target_ids
        matching_ids = source_ids & target_ids
        orphaned_in_target = target_ids - source_ids  # We don't modify these

        self.stats.missing_in_target = len(missing_in_target)
        self.stats.matching_ids = len(matching_ids)

        console.print(f"  [yellow]Missing in target (to migrate):[/yellow] {len(missing_in_target)}")
        console.print(f"  [cyan]Matching IDs (to compare):[/cyan] {len(matching_ids)}")
        console.print(f"  [dim]Orphaned in target (ignored):[/dim] {len(orphaned_in_target)}")

        # Step 4: Compare metadata for matching IDs
        if matching_ids:
            console.print("\n[bold cyan]Step 4/4: Comparing metadata for matching vectors...[/bold cyan]")
            self._analyze_metadata_differences(
                list(matching_ids),
                namespace,
                sample_size,
                show_progress
            )
        else:
            console.print("\n[bold cyan]Step 4/4: No matching vectors to compare[/bold cyan]")

        # Display analysis summary
        self._display_analysis_summary(namespace)

        return self.stats

    def _analyze_metadata_differences(
        self,
        matching_ids: List[str],
        namespace: str,
        sample_size: int,
        show_progress: bool
    ):
        """
        Analyze metadata differences for matching vector IDs.

        Fetches metadata from both indexes and compares them.
        """
        batch_size = 100
        total_batches = (len(matching_ids) + batch_size - 1) // batch_size

        samples_collected = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=not show_progress
        ) as progress:
            task = progress.add_task(
                "Comparing metadata...",
                total=len(matching_ids)
            )

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(matching_ids))
                batch_ids = matching_ids[start_idx:end_idx]

                # Fetch from both indexes
                source_vectors = self.pinecone.fetch_vectors_by_ids(
                    batch_ids, namespace, source=True, show_progress=False
                )
                target_vectors = self.pinecone.fetch_vectors_by_ids(
                    batch_ids, namespace, source=False, show_progress=False
                )

                # Create lookup dicts
                source_dict = {v.id: v for v in source_vectors}
                target_dict = {v.id: v for v in target_vectors}

                # Compare each matching pair
                for vec_id in batch_ids:
                    source_vec = source_dict.get(vec_id)
                    target_vec = target_dict.get(vec_id)

                    if source_vec and target_vec:
                        differs, diff_keys = compare_metadata(
                            source_vec.metadata,
                            target_vec.metadata
                        )

                        if differs:
                            self.stats.metadata_differs += 1

                            # Collect sample for reporting
                            if samples_collected < sample_size:
                                self.stats.metadata_diff_details.append({
                                    'id': vec_id,
                                    'diff_keys': diff_keys,
                                    'source_meta_keys': list(source_vec.metadata.keys()),
                                    'target_meta_keys': list(target_vec.metadata.keys())
                                })
                                samples_collected += 1
                        else:
                            self.stats.metadata_matches += 1

                progress.update(task, advance=len(batch_ids))

        console.print(f"\n  [green]Metadata matches (no action):[/green] {self.stats.metadata_matches}")
        console.print(f"  [yellow]Metadata differs (to update):[/yellow] {self.stats.metadata_differs}")

    def _display_analysis_summary(self, namespace: str):
        """Display a summary table of the analysis results"""
        table = Table(title=f"Analysis Summary: {namespace}")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="yellow")
        table.add_column("Action", style="green")

        table.add_row(
            "Total Source Vectors",
            str(self.stats.total_source_vectors),
            ""
        )
        table.add_row(
            "Total Target Vectors",
            str(self.stats.total_target_vectors),
            ""
        )
        table.add_row("─" * 20, "─" * 10, "─" * 20)
        table.add_row(
            "Missing in Target",
            str(self.stats.missing_in_target),
            "[yellow]Will be migrated[/yellow]" if self.stats.missing_in_target > 0 else "[green]None[/green]"
        )
        table.add_row(
            "Matching IDs",
            str(self.stats.matching_ids),
            "[cyan]Compared metadata[/cyan]"
        )
        table.add_row(
            "  └─ Metadata Matches",
            str(self.stats.metadata_matches),
            "[green]No action needed[/green]"
        )
        table.add_row(
            "  └─ Metadata Differs",
            str(self.stats.metadata_differs),
            "[yellow]Will be updated[/yellow]" if self.stats.metadata_differs > 0 else "[green]None[/green]"
        )

        console.print("\n")
        console.print(table)

        # Show sample of differences if any
        if self.stats.metadata_diff_details:
            console.print("\n[bold]Sample Metadata Differences:[/bold]")
            for detail in self.stats.metadata_diff_details[:3]:
                console.print(f"  [cyan]ID:[/cyan] {detail['id'][:20]}...")
                console.print(f"  [yellow]Changed keys:[/yellow] {', '.join(detail['diff_keys'])}")
                console.print()

        # Total actions needed
        total_actions = self.stats.missing_in_target + self.stats.metadata_differs
        if total_actions > 0:
            console.print(Panel(
                f"[bold yellow]Total vectors requiring action: {total_actions}[/bold yellow]\n"
                f"  • Missing (to migrate): {self.stats.missing_in_target}\n"
                f"  • Different (to update): {self.stats.metadata_differs}",
                title="Action Required",
                border_style="yellow"
            ))
        else:
            console.print(Panel(
                "[bold green]Namespace is fully synchronized![/bold green]\n"
                "No migration or updates needed.",
                title="In Sync",
                border_style="green"
            ))

    def mirror_namespace(
        self,
        namespace: str,
        dry_run: bool = False,
        batch_size: int = 100,
        skip_analysis: bool = False
    ) -> MirroringStats:
        """
        Perform full namespace mirroring.

        This operation:
        1. Analyzes the namespace (unless skip_analysis=True)
        2. Migrates missing vectors (Phase A)
        3. Updates vectors with metadata differences (Phase B)

        Args:
            namespace: Namespace to mirror
            dry_run: If True, don't make changes, just report what would be done
            batch_size: Number of vectors to process per batch
            skip_analysis: If True, skip analysis (use previous stats)

        Returns:
            MirroringStats with operation results
        """
        start_time = datetime.now()

        console.print(Panel(
            f"[bold]Starting namespace mirroring:[/bold] {namespace}\n"
            f"[dim]Mode: {'DRY RUN' if dry_run else 'LIVE'}[/dim]",
            title="Mirroring Started",
            border_style="blue"
        ))

        # Step 1: Analyze namespace
        if not skip_analysis:
            console.print("\n[bold magenta]═══ ANALYSIS PHASE ═══[/bold magenta]")
            self.analyze_namespace(namespace, show_progress=True)

        # Check if any work to do
        if self.stats.missing_in_target == 0 and self.stats.metadata_differs == 0:
            console.print("\n[green]Namespace is already in sync. No changes needed.[/green]")
            return self.stats

        if dry_run:
            console.print("\n[yellow]DRY RUN MODE - No changes will be made[/yellow]")

        # Step 2: Process missing vectors (Phase A - Migration)
        if self.stats.missing_in_target > 0:
            console.print("\n[bold magenta]═══ PHASE A: MIGRATING MISSING VECTORS ═══[/bold magenta]")
            self._process_missing_vectors(namespace, dry_run, batch_size)

        # Step 3: Process metadata updates (Phase B - Sync)
        if self.stats.metadata_differs > 0:
            console.print("\n[bold magenta]═══ PHASE B: UPDATING CHANGED VECTORS ═══[/bold magenta]")
            self._process_metadata_updates(namespace, dry_run, batch_size)

        # Calculate duration
        duration = datetime.now() - start_time

        # Display final results
        self._display_final_stats(namespace, duration, dry_run)

        return self.stats

    def _process_missing_vectors(
        self,
        namespace: str,
        dry_run: bool,
        batch_size: int
    ):
        """
        Process vectors that are missing in target index.

        This is similar to the original migration logic:
        1. Fetch full vectors from source
        2. Extract text and generate new embeddings
        3. Add clearance_level metadata
        4. Upsert to target
        """
        console.print(f"[cyan]Processing {self.stats.missing_in_target} missing vectors...[/cyan]")

        # Get IDs to migrate
        source_ids = self.pinecone.get_all_vector_ids(namespace, source=True)
        target_ids = self.pinecone.get_all_vector_ids(namespace, source=False)
        missing_ids = list(source_ids - target_ids)

        if not missing_ids:
            console.print("[green]No missing vectors found[/green]")
            return

        # Fetch vectors to migrate
        console.print(f"[cyan]Fetching {len(missing_ids)} vectors from source...[/cyan]")
        vectors_to_migrate = self.pinecone.fetch_vectors_by_ids(
            missing_ids, namespace, source=True
        )

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
                "Migrating missing vectors...",
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

                if not texts:
                    progress.update(task, advance=len(batch))
                    continue

                # Generate new embeddings
                try:
                    embeddings = self.gemini.generate_batch_embeddings(texts, show_progress=False)
                except Exception as e:
                    console.print(f"[red]Error generating embeddings:[/red] {e}")
                    self.stats.failed_migration += len(valid_vectors)
                    progress.update(task, advance=len(batch))
                    continue

                # Prepare vectors for upsert
                vectors_to_upsert = []
                for vec, embedding in zip(valid_vectors, embeddings):
                    if embedding is not None:
                        new_metadata = dict(vec.metadata)
                        new_metadata[config.migration.new_metadata_key] = config.migration.new_metadata_value

                        vectors_to_upsert.append({
                            "id": vec.id,
                            "values": embedding,
                            "metadata": new_metadata
                        })
                    else:
                        self.stats.failed_migration += 1

                # Upsert to target
                if vectors_to_upsert and not dry_run:
                    try:
                        upserted = self.pinecone.upsert_vectors(vectors_to_upsert, namespace)
                        self.stats.successfully_migrated += upserted
                    except Exception as e:
                        console.print(f"[red]Error upserting:[/red] {e}")
                        self.stats.failed_migration += len(vectors_to_upsert)
                elif vectors_to_upsert and dry_run:
                    self.stats.successfully_migrated += len(vectors_to_upsert)

                progress.update(task, advance=len(batch))

                # Delay between batches
                if batch_idx < total_batches - 1:
                    time.sleep(config.migration.batch_delay)

        console.print(f"[green]Migration complete: {self.stats.successfully_migrated} vectors[/green]")

    def _process_metadata_updates(
        self,
        namespace: str,
        dry_run: bool,
        batch_size: int
    ):
        """
        Process vectors with metadata differences.

        For vectors that exist in both indexes but have different metadata:
        1. Fetch full vectors from source
        2. Extract text and generate new embeddings
        3. Preserve source metadata + add clearance_level
        4. Upsert to target (REPLACES existing vector)
        """
        console.print(f"[cyan]Identifying vectors with metadata differences...[/cyan]")

        # Get matching IDs and identify those with differences
        source_ids = self.pinecone.get_all_vector_ids(namespace, source=True)
        target_ids = self.pinecone.get_all_vector_ids(namespace, source=False)
        matching_ids = list(source_ids & target_ids)

        if not matching_ids:
            console.print("[green]No matching vectors to update[/green]")
            return

        # Find IDs with metadata differences
        ids_to_update = []
        check_batch_size = 100

        console.print(f"[cyan]Checking {len(matching_ids)} matching vectors for differences...[/cyan]")

        for i in range(0, len(matching_ids), check_batch_size):
            batch_ids = matching_ids[i:i + check_batch_size]

            source_vectors = self.pinecone.fetch_vectors_by_ids(
                batch_ids, namespace, source=True, show_progress=False
            )
            target_vectors = self.pinecone.fetch_vectors_by_ids(
                batch_ids, namespace, source=False, show_progress=False
            )

            source_dict = {v.id: v for v in source_vectors}
            target_dict = {v.id: v for v in target_vectors}

            for vec_id in batch_ids:
                source_vec = source_dict.get(vec_id)
                target_vec = target_dict.get(vec_id)

                if source_vec and target_vec:
                    differs, _ = compare_metadata(source_vec.metadata, target_vec.metadata)
                    if differs:
                        ids_to_update.append(vec_id)

        if not ids_to_update:
            console.print("[green]No vectors need updating[/green]")
            return

        console.print(f"[yellow]Found {len(ids_to_update)} vectors to update[/yellow]")

        # Fetch source vectors to update
        vectors_to_update = self.pinecone.fetch_vectors_by_ids(
            ids_to_update, namespace, source=True
        )

        # Process in batches
        total_batches = (len(vectors_to_update) + batch_size - 1) // batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "Updating changed vectors...",
                total=len(vectors_to_update)
            )

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(vectors_to_update))
                batch = vectors_to_update[start_idx:end_idx]

                # Extract texts for re-embedding
                texts = []
                valid_vectors = []
                for vec in batch:
                    text = vec.metadata.get("text", "")
                    if text and text.strip():
                        texts.append(text)
                        valid_vectors.append(vec)
                    else:
                        self.stats.skipped_no_text += 1

                if not texts:
                    progress.update(task, advance=len(batch))
                    continue

                # Generate new embeddings
                try:
                    embeddings = self.gemini.generate_batch_embeddings(texts, show_progress=False)
                except Exception as e:
                    console.print(f"[red]Error generating embeddings:[/red] {e}")
                    self.stats.failed_update += len(valid_vectors)
                    progress.update(task, advance=len(batch))
                    continue

                # Prepare vectors for upsert (this will REPLACE existing)
                vectors_to_upsert = []
                for vec, embedding in zip(valid_vectors, embeddings):
                    if embedding is not None:
                        # Use source metadata + add clearance_level
                        new_metadata = dict(vec.metadata)
                        new_metadata[config.migration.new_metadata_key] = config.migration.new_metadata_value

                        vectors_to_upsert.append({
                            "id": vec.id,
                            "values": embedding,
                            "metadata": new_metadata
                        })
                    else:
                        self.stats.failed_update += 1

                # Upsert to target (REPLACES existing vectors)
                if vectors_to_upsert and not dry_run:
                    try:
                        upserted = self.pinecone.upsert_vectors(vectors_to_upsert, namespace)
                        self.stats.successfully_updated += upserted
                    except Exception as e:
                        console.print(f"[red]Error upserting:[/red] {e}")
                        self.stats.failed_update += len(vectors_to_upsert)
                elif vectors_to_upsert and dry_run:
                    self.stats.successfully_updated += len(vectors_to_upsert)

                progress.update(task, advance=len(batch))

                # Delay between batches
                if batch_idx < total_batches - 1:
                    time.sleep(config.migration.batch_delay)

        console.print(f"[green]Update complete: {self.stats.successfully_updated} vectors[/green]")

    def _display_final_stats(self, namespace: str, duration, dry_run: bool):
        """Display final mirroring statistics"""
        mode_suffix = " (Dry Run)" if dry_run else ""

        table = Table(title=f"Mirroring Results: {namespace}{mode_suffix}")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="yellow")

        # Analysis section
        table.add_row("─── Analysis ───", "")
        table.add_row("Total Source Vectors", str(self.stats.total_source_vectors))
        table.add_row("Total Target Vectors", str(self.stats.total_target_vectors))
        table.add_row("Missing in Target", str(self.stats.missing_in_target))
        table.add_row("Metadata Differs", str(self.stats.metadata_differs))

        # Results section
        table.add_row("─── Results ───", "")
        table.add_row(
            "Successfully Migrated",
            f"[green]{self.stats.successfully_migrated}[/green]"
        )
        table.add_row(
            "Successfully Updated",
            f"[green]{self.stats.successfully_updated}[/green]"
        )
        table.add_row(
            "Failed (Migration)",
            f"[red]{self.stats.failed_migration}[/red]" if self.stats.failed_migration > 0 else "0"
        )
        table.add_row(
            "Failed (Update)",
            f"[red]{self.stats.failed_update}[/red]" if self.stats.failed_update > 0 else "0"
        )
        table.add_row(
            "Skipped (No Text)",
            f"[yellow]{self.stats.skipped_no_text}[/yellow]" if self.stats.skipped_no_text > 0 else "0"
        )
        table.add_row("Duration", str(duration))

        console.print("\n")
        console.print(table)

        # Summary
        total_processed = self.stats.successfully_migrated + self.stats.successfully_updated
        total_failed = self.stats.failed_migration + self.stats.failed_update

        if total_failed == 0 and total_processed > 0:
            console.print(Panel(
                f"[bold green]Mirroring completed successfully![/bold green]\n"
                f"Processed {total_processed} vectors without errors.",
                title="Success",
                border_style="green"
            ))
        elif total_failed > 0:
            console.print(Panel(
                f"[bold yellow]Mirroring completed with some failures[/bold yellow]\n"
                f"Processed: {total_processed}, Failed: {total_failed}",
                title="Partial Success",
                border_style="yellow"
            ))
        else:
            console.print(Panel(
                "[bold green]No changes were needed - namespace is in sync![/bold green]",
                title="Already Synced",
                border_style="green"
            ))

        # Verification
        if not dry_run and total_processed > 0:
            console.print("\n[bold cyan]Verifying synchronization...[/bold cyan]")

            source_count = len(self.pinecone.get_all_vector_ids(namespace, source=True))
            target_count = len(self.pinecone.get_all_vector_ids(namespace, source=False))

            verification_table = Table(title="Verification")
            verification_table.add_column("Index", style="cyan")
            verification_table.add_column("Vector Count", style="yellow")

            verification_table.add_row("Source", str(source_count))
            verification_table.add_row("Target", str(target_count))

            console.print(verification_table)

            if source_count == target_count:
                console.print("[green]✓ Vector counts match![/green]")
            else:
                diff = source_count - target_count
                console.print(f"[yellow]⚠ Difference: {diff} vectors[/yellow]")


# Global service instance
mirroring_service = MirroringService()
