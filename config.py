"""
Configuration for Pinecone Vector Embedding Migration

Supports both environment variables and hardcoded defaults.
To use environment variables, create a .env file based on .env.example
"""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class PineconeConfig:
    """Pinecone configuration"""
    api_key: str
    old_index_name: str = "gflops-serverless"
    new_index_name: str = "askdona"
    environment: str = "us-west-2"


@dataclass
class GeminiConfig:
    """Gemini embedding configuration"""
    api_key: str
    model: str = "gemini-embedding-001"  # Gemini embedding model with 3072 dimensions
    output_dimensions: int = 3072
    task_type: str = "RETRIEVAL_DOCUMENT"


@dataclass
class ElasticsearchConfig:
    """Elasticsearch configuration"""
    endpoint: str
    api_key: str
    index_name: str = "text-search-v0"


@dataclass
class MigrationConfig:
    """Migration configuration"""
    # Batch sizes
    query_batch_size: int = 100  # How many vectors to query at once from source
    embedding_batch_size: int = 100  # How many texts to embed at once
    upsert_batch_size: int = 100  # How many vectors to upsert at once

    # Delays (in seconds)
    batch_delay: float = 0.5  # Delay between batches to avoid rate limits

    # New metadata to add
    new_metadata_key: str = "clearance_level"
    new_metadata_value: int = 1  # Must be integer, not string

    # Default namespace to migrate
    default_namespace: str = "51c04445-7c02-40a8-bb6b-fbaaa6b0000e"


class Config:
    """Main configuration class"""

    def __init__(self):
        # Load from environment variables (required)
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        google_api_key = os.getenv("GOOGLE_AI_API_KEY")
        elasticsearch_endpoint = os.getenv("ELASTICSEARCH_ENDPOINT")
        elasticsearch_api_key = os.getenv("ELASTICSEARCH_API_KEY")

        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required. Copy .env.example to .env and fill in your API key.")
        if not google_api_key:
            raise ValueError("GOOGLE_AI_API_KEY environment variable is required. Copy .env.example to .env and fill in your API key.")
        if not elasticsearch_endpoint:
            raise ValueError("ELASTICSEARCH_ENDPOINT environment variable is required. Copy .env.example to .env and fill in your endpoint.")
        if not elasticsearch_api_key:
            raise ValueError("ELASTICSEARCH_API_KEY environment variable is required. Copy .env.example to .env and fill in your API key.")

        self.pinecone = PineconeConfig(
            api_key=pinecone_api_key,
            old_index_name=os.getenv("PINECONE_OLD_INDEX", "gflops-serverless"),
            new_index_name=os.getenv("PINECONE_NEW_INDEX", "askdona"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west-2")
        )

        self.gemini = GeminiConfig(
            api_key=google_api_key,
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            output_dimensions=int(os.getenv("GEMINI_OUTPUT_DIMENSIONS", "3072"))
        )

        self.migration = MigrationConfig(
            query_batch_size=int(os.getenv("QUERY_BATCH_SIZE", "100")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            upsert_batch_size=int(os.getenv("UPSERT_BATCH_SIZE", "100")),
            batch_delay=float(os.getenv("BATCH_DELAY", "0.5")),
            new_metadata_key=os.getenv("NEW_METADATA_KEY", "clearance_level"),
            new_metadata_value=int(os.getenv("NEW_METADATA_VALUE", "1")),  # Convert to int
            default_namespace=os.getenv("DEFAULT_NAMESPACE", "51c04445-7c02-40a8-bb6b-fbaaa6b0000e")
        )

        self.elasticsearch = ElasticsearchConfig(
            endpoint=elasticsearch_endpoint,
            api_key=elasticsearch_api_key,
            index_name=os.getenv("ELASTICSEARCH_INDEX", "text-search-v0")
        )

    def display_info(self) -> str:
        """Return configuration info for display"""
        return f"""
================================================================================
                    PINECONE MIGRATION CONFIGURATION
================================================================================

SOURCE INDEX (READ-ONLY):
  - Name: {self.pinecone.old_index_name}
  - Environment: {self.pinecone.environment}

TARGET INDEX:
  - Name: {self.pinecone.new_index_name}
  - Environment: {self.pinecone.environment}

GEMINI EMBEDDING:
  - Model: {self.gemini.model}
  - Dimensions: {self.gemini.output_dimensions}
  - Task Type: {self.gemini.task_type}

MIGRATION SETTINGS:
  - Query Batch Size: {self.migration.query_batch_size}
  - Embedding Batch Size: {self.migration.embedding_batch_size}
  - Upsert Batch Size: {self.migration.upsert_batch_size}
  - Batch Delay: {self.migration.batch_delay}s

NEW METADATA ADDED TO ALL EMBEDDINGS:
  ┌────────────────────────────────────────────────────────┐
  │  Key: {self.migration.new_metadata_key:<20}                        │
  │  Value: {self.migration.new_metadata_value:<18}                        │
  └────────────────────────────────────────────────────────┘

DEFAULT NAMESPACE: {self.migration.default_namespace}

================================================================================
"""


# Global config instance
config = Config()
