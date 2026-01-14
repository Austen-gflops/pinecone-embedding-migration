# Pinecone Vector Embedding Migration Tool

This Python application migrates vector embeddings from an old Pinecone index (`gflops-serverless`) to a new index (`askdona`) using Gemini embeddings.

## Features

- **Check Namespace Counts**: View embedding counts for any namespace
- **Compare Indexes**: See the difference between source and target namespaces
- **List Namespaces**: Browse all namespaces in an index
- **Idempotent Migration**: Automatically skips already-migrated vectors
- **Batch Processing**: Efficiently handles thousands of vectors
- **Metadata Preservation**: Keeps all original metadata intact
- **New Metadata Addition**: Adds `clearance_level: "1"` to all embeddings

## Migration Details

| Property | Source (Old) | Target (New) |
|----------|--------------|--------------|
| Index Name | `gflops-serverless` | `askdona` |
| Embedding Model | OpenAI | Gemini (`gemini-embedding-001`) |
| Dimensions | Variable | 3072 |

### New Metadata Added

Every migrated embedding will have a new metadata field:

```json
{
  "clearance_level": "1"
}
```

This is added **in addition to** all existing metadata.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip3

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Austen-gflops/pinecone-embedding-migration.git
   cd pinecone-embedding-migration
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Quick Start

```bash
source venv/bin/activate
python3 main.py
```

Then select option `7` for quick migration of the default namespace.

## Usage

### Start the Application

```bash
source venv/bin/activate
python3 main.py
```

### Menu Options

The application provides an interactive menu:

1. **Check namespace embedding count** - View vector counts for a specific namespace
2. **Compare namespaces between indexes** - See source vs target comparison
3. **List all namespaces in an index** - Browse available namespaces
4. **Run migration for a namespace** - Execute the actual migration
5. **Run migration (dry run)** - Preview what would be migrated without making changes
6. **Show configuration** - Display current settings
7. **Quick migrate default namespace** - One-click migration for the default namespace

### Quick Migration

For the namespace `51c04445-7c02-40a8-bb6b-fbaaa6b0000e`:

1. Start the application: `python3 main.py`
2. Press `7` for Quick Migrate
3. Confirm the operation

### Checking Namespace Counts

1. Start the application: `python3 main.py`
2. Press `1` for Check namespace embedding count
3. Enter the namespace ID (or press Enter for default)
4. Choose to check source, target, or both indexes

## Safety Features

- **Read-Only Source**: The source index (`gflops-serverless`) is never modified
- **No Deletions**: No vectors are ever deleted from either index
- **Idempotent**: Running migration multiple times is safe - already migrated vectors are skipped
- **Dry Run Mode**: Test the migration without making any changes

## Configuration

Configuration is stored in `config.py`. Key settings:

```python
# Pinecone Indexes
OLD_INDEX = "gflops-serverless"  # Source (read-only)
NEW_INDEX = "askdona"            # Target

# Gemini Embedding
MODEL = "gemini-embedding-001"   # Gemini embedding model
DIMENSIONS = 3072                # Output dimensions
TASK_TYPE = "RETRIEVAL_DOCUMENT"

# Batch Processing
QUERY_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 100
BATCH_DELAY = 0.5  # seconds

# New Metadata
NEW_METADATA_KEY = "clearance_level"
NEW_METADATA_VALUE = "1"
```

## Migration Workflow

```
1. FETCH from source (gflops-serverless)
   - Query all vector IDs in namespace
   - Fetch metadata for each vector

2. CHECK target (askdona)
   - Get existing vector IDs
   - Skip vectors already migrated

3. GENERATE embeddings
   - Extract text from metadata
   - Generate Gemini embeddings in batches
   - 3072 dimensions output

4. PREPARE vectors
   - Keep original ID
   - Keep all original metadata
   - Add clearance_level = "1"

5. UPSERT to target
   - Batch upsert to askdona
   - Same namespace as source
```

## Troubleshooting

### Rate Limits

If you encounter rate limit errors:
- The application automatically retries with exponential backoff
- You can increase `BATCH_DELAY` in `config.py`

### Missing Text

Vectors without `text` in metadata are skipped with a warning.

### Connection Issues

Ensure you have internet access and the API keys are valid.

## API Keys

The application requires the following API keys (set in `.env` file):

| Variable | Source |
|----------|--------|
| `PINECONE_API_KEY` | [Pinecone Console](https://app.pinecone.io/) |
| `GOOGLE_AI_API_KEY` | [Google AI Studio](https://makersuite.google.com/app/apikey) |

Copy `.env.example` to `.env` and fill in your API keys.

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI application entry point |
| `config.py` | Configuration settings |
| `pinecone_client.py` | Pinecone API client |
| `gemini_client.py` | Gemini embedding client |
| `migration_service.py` | Core migration logic |
| `requirements.txt` | Python dependencies |

## License

MIT License
