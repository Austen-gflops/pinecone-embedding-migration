"""
Gemini Embedding Client for batch embedding generation using Google GenAI SDK
"""
from typing import List, Optional
import time

from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config

console = Console()


class GeminiEmbeddingClient:
    """Client for generating embeddings using Gemini API"""

    def __init__(self):
        self.client = genai.Client(api_key=config.gemini.api_key)
        self.model = config.gemini.model
        self.output_dimensions = config.gemini.output_dimensions
        console.print(f"[green]Initialized Gemini client with model:[/green] {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic"""
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensions
            )
        )
        return result.embeddings[0].values

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        try:
            if not text or not text.strip():
                console.print("[yellow]Warning: Empty text provided, skipping[/yellow]")
                return None

            embedding = self._generate_single_embedding(text)

            if len(embedding) != self.output_dimensions:
                console.print(f"[yellow]Warning: Expected {self.output_dimensions} dimensions, got {len(embedding)}[/yellow]")

            return embedding

        except Exception as e:
            console.print(f"[red]Error generating embedding:[/red] {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with retry logic"""
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensions
            )
        )
        return [emb.values for emb in result.embeddings]

    def generate_batch_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embeddings (None for failed texts)
        """
        if not texts:
            return []

        batch_size = config.migration.embedding_batch_size
        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        # Filter out empty texts but track their positions
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            console.print("[yellow]No valid texts to embed[/yellow]")
            return all_embeddings

        if show_progress:
            console.print(f"[cyan]Generating embeddings for {len(valid_texts)} texts in batches of {batch_size}...[/cyan]")

        total_batches = (len(valid_texts) + batch_size - 1) // batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            disable=not show_progress
        ) as progress:
            task = progress.add_task("Generating embeddings...", total=len(valid_texts))

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(valid_texts))
                batch_texts = valid_texts[start_idx:end_idx]

                try:
                    batch_embeddings = self._generate_batch_embeddings(batch_texts)

                    # Map back to original positions
                    for i, embedding in enumerate(batch_embeddings):
                        original_idx = valid_indices[start_idx + i]
                        all_embeddings[original_idx] = embedding

                    progress.update(task, advance=len(batch_texts))

                except Exception as e:
                    console.print(f"[red]Error in batch {batch_idx + 1}:[/red] {e}")
                    # Try individual embeddings for failed batch
                    for i, text in enumerate(batch_texts):
                        try:
                            embedding = self._generate_single_embedding(text)
                            original_idx = valid_indices[start_idx + i]
                            all_embeddings[original_idx] = embedding
                        except Exception as inner_e:
                            console.print(f"[red]Failed to embed text at index {start_idx + i}:[/red] {inner_e}")

                        progress.update(task, advance=1)

                # Delay between batches to avoid rate limits
                if batch_idx < total_batches - 1:
                    time.sleep(config.migration.batch_delay)

        # Count successful embeddings
        successful = sum(1 for e in all_embeddings if e is not None)
        if show_progress:
            console.print(f"[green]Successfully generated {successful}/{len(texts)} embeddings[/green]")

        return all_embeddings


# Global client instance
gemini_client = GeminiEmbeddingClient()
