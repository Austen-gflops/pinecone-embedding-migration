#!/usr/bin/env python3
"""
Pinecone Vector Embedding Migration Tool

This tool migrates vector embeddings from the old Pinecone index (gflops-serverless)
to the new index (askdona) using Gemini embeddings.

Features:
- Check namespace embedding counts
- Idempotent migration (skips already migrated vectors)
- Batch processing for efficiency
- Adds new metadata (clearance_level: "1") to all embeddings
- Preserves original IDs and metadata
"""
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from config import config
from pinecone_client import pinecone_client
from migration_service import migration_service

console = Console()


def display_welcome():
    """Display welcome message and configuration"""
    console.print(Panel(
        """[bold cyan]Pinecone Vector Embedding Migration Tool[/bold cyan]

Migrate embeddings from OpenAI to Gemini format.

[yellow]IMPORTANT:[/yellow]
- Source index (gflops-serverless) is READ-ONLY
- No vectors will be deleted from source
- All original metadata is preserved
- New metadata 'clearance_level: 1' is added""",
        title="Welcome",
        border_style="blue"
    ))


def display_menu():
    """Display main menu options"""
    console.print("\n[bold cyan]Main Menu:[/bold cyan]")
    console.print("  [1] Check namespace embedding count")
    console.print("  [2] Compare namespaces between indexes")
    console.print("  [3] List all namespaces in an index")
    console.print("  [4] Run migration for a namespace")
    console.print("  [5] Run migration (dry run)")
    console.print("  [6] Show configuration")
    console.print("  [7] Quick migrate default namespace")
    console.print("  [0] Exit")
    console.print()


def check_namespace_count():
    """Check embedding count for a specific namespace"""
    console.print("\n[bold]Check Namespace Embedding Count[/bold]\n")

    namespace = Prompt.ask(
        "Enter namespace",
        default=config.migration.default_namespace
    )

    index_choice = Prompt.ask(
        "Which index?",
        choices=["source", "target", "both"],
        default="both"
    )

    if index_choice in ["source", "both"]:
        source_stats = pinecone_client.get_namespace_stats(
            config.pinecone.old_index_name, namespace
        )
        console.print(f"\n[cyan]Source ({config.pinecone.old_index_name}):[/cyan]")
        console.print(f"  Vector Count: [yellow]{source_stats.get('vector_count', 0)}[/yellow]")
        console.print(f"  Exists: {'[green]Yes[/green]' if source_stats.get('exists') else '[red]No[/red]'}")

    if index_choice in ["target", "both"]:
        target_stats = pinecone_client.get_namespace_stats(
            config.pinecone.new_index_name, namespace
        )
        console.print(f"\n[cyan]Target ({config.pinecone.new_index_name}):[/cyan]")
        console.print(f"  Vector Count: [yellow]{target_stats.get('vector_count', 0)}[/yellow]")
        console.print(f"  Exists: {'[green]Yes[/green]' if target_stats.get('exists') else '[red]No[/red]'}")


def compare_namespaces():
    """Compare namespace between source and target indexes"""
    console.print("\n[bold]Compare Namespaces Between Indexes[/bold]\n")

    namespace = Prompt.ask(
        "Enter namespace",
        default=config.migration.default_namespace
    )

    migration_service.display_namespace_comparison(namespace)


def list_namespaces():
    """List all namespaces in an index"""
    console.print("\n[bold]List All Namespaces[/bold]\n")

    index_choice = Prompt.ask(
        "Which index?",
        choices=["source", "target"],
        default="source"
    )

    index_name = (
        config.pinecone.old_index_name if index_choice == "source"
        else config.pinecone.new_index_name
    )

    console.print(f"\n[cyan]Fetching namespaces from {index_name}...[/cyan]")

    namespaces = pinecone_client.list_all_namespaces(index_name)

    if namespaces:
        table = Table(title=f"Namespaces in {index_name}")
        table.add_column("#", style="dim")
        table.add_column("Namespace", style="cyan")
        table.add_column("Vector Count", style="yellow")

        for i, ns in enumerate(namespaces, 1):
            stats = pinecone_client.get_namespace_stats(index_name, ns)
            table.add_row(
                str(i),
                ns,
                str(stats.get("vector_count", 0))
            )

        console.print(table)
    else:
        console.print("[yellow]No namespaces found[/yellow]")


def run_migration(dry_run: bool = False):
    """Run the migration process"""
    console.print(f"\n[bold]Run Migration {'(Dry Run)' if dry_run else ''}[/bold]\n")

    namespace = Prompt.ask(
        "Enter namespace to migrate",
        default=config.migration.default_namespace
    )

    # Show current state
    migration_service.display_namespace_comparison(namespace)

    # Confirm
    if not dry_run:
        console.print("\n[yellow]WARNING: This will upsert new vectors to the target index.[/yellow]")
        console.print("[green]Source index will NOT be modified (read-only).[/green]")

    if not Confirm.ask("\nProceed with migration?"):
        console.print("[dim]Migration cancelled[/dim]")
        return

    # Run migration
    stats = migration_service.migrate_namespace(namespace, dry_run=dry_run)

    console.print("\n[bold green]Migration complete![/bold green]")


def quick_migrate():
    """Quick migrate the default namespace"""
    console.print("\n[bold]Quick Migration - Default Namespace[/bold]\n")

    namespace = config.migration.default_namespace
    console.print(f"[cyan]Namespace:[/cyan] {namespace}")

    # Show current state
    migration_service.display_namespace_comparison(namespace)

    console.print("\n[yellow]This will migrate all pending vectors.[/yellow]")
    console.print("[green]Already migrated vectors will be skipped.[/green]")
    console.print("[green]Source index will NOT be modified.[/green]")

    if not Confirm.ask("\nProceed with migration?"):
        console.print("[dim]Migration cancelled[/dim]")
        return

    # Run migration
    stats = migration_service.migrate_namespace(namespace, dry_run=False)

    console.print("\n[bold green]Migration complete![/bold green]")


def show_configuration():
    """Show current configuration"""
    migration_service.display_migration_info()


def main():
    """Main entry point"""
    display_welcome()

    while True:
        display_menu()

        try:
            choice = Prompt.ask("Select option", default="0")

            if choice == "0":
                console.print("\n[dim]Goodbye![/dim]")
                break
            elif choice == "1":
                check_namespace_count()
            elif choice == "2":
                compare_namespaces()
            elif choice == "3":
                list_namespaces()
            elif choice == "4":
                run_migration(dry_run=False)
            elif choice == "5":
                run_migration(dry_run=True)
            elif choice == "6":
                show_configuration()
            elif choice == "7":
                quick_migrate()
            else:
                console.print("[red]Invalid option. Please try again.[/red]")

        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            if Confirm.ask("Continue?", default=True):
                continue
            break


if __name__ == "__main__":
    main()
