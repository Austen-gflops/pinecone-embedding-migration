# Pinecone Vector Embedding Migration Tool

A Python application with **Web GUI** for migrating vector embeddings from an old Pinecone index (`gflops-serverless`) to a new index (`askdona`) using Gemini embeddings.

## Features

- **Web-Based GUI**: Beautiful browser interface powered by Streamlit
- **Dashboard**: Overview of migration status and statistics
- **Check Namespace Counts**: View embedding counts for any namespace
- **Namespace Mirroring**: Continuous synchronization between indexes (NEW!)
- **Idempotent Migration**: Automatically skips already-migrated vectors
- **Metadata Change Detection**: Identifies and updates vectors with changed metadata
- **Batch Processing**: Efficiently handles thousands of vectors
- **Metadata Preservation**: Keeps all original metadata intact
- **New Metadata Addition**: Adds `clearance_level: 1` (integer) to all embeddings
- **Progress Tracking**: Real-time progress bar during migration
- **Dry Run Mode**: Preview migration without making changes

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
  "clearance_level": 1
}
```

**Note:** The value is an integer (`1`), not a string (`"1"`).

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

### Web GUI (Recommended)

Launch the web interface:

```bash
source venv/bin/activate
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### CLI Interface

For command-line usage:

```bash
source venv/bin/activate
python3 main.py
```

## Web GUI Pages

### 📊 Dashboard
- Overview of source and target indexes
- Migration progress visualization
- Elasticsearch status and progress
- Quick statistics

### 🔍 Check Namespace
- Enter any namespace ID
- View vector counts in source, target, or both indexes

### 🪞 Namespace Mirroring (NEW!)
- **Continuous synchronization** between source and target indexes
- **Analyze mode**: Preview what needs to be synchronized
- **Two-phase operation**:
  - Phase A: Migrate missing vectors (source → target)
  - Phase B: Update vectors with metadata differences
- **Metadata comparison**: Detects changes in existing vectors
- **Dry run support**: Preview changes before applying

### 🚀 Pinecone ReEmbedding
- One-time migration of vectors
- Configure batch size
- Enable/disable dry run mode
- Real-time progress tracking
- Detailed migration log

### 🔎 Elasticsearch Update
- Add `clearance_level` to Elasticsearch documents
- Filter by namespace
- Only update missing or update all
- Bulk processing with progress tracking

### ⚙️ Configuration
- View all current settings
- Pinecone, Gemini, Elasticsearch, and migration configuration

## Namespace Mirroring

The **Namespace Mirroring** feature ensures continuous synchronization between your source and target indexes. This is particularly useful when:

- Users upload new files (new vectors to migrate)
- Users update metadata for existing records (vectors need re-embedding)
- You need to run sync operations on a regular basis

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    MIRRORING WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

1. ANALYZE
   ├─ Get all IDs from source index
   ├─ Get all IDs from target index
   ├─ Categorize:
   │   ├─ Missing in target → To migrate (Phase A)
   │   └─ Exist in both → Compare metadata
   │       ├─ Metadata matches → No action
   │       └─ Metadata differs → To update (Phase B)
   └─ Report analysis results

2. PHASE A: MIGRATE MISSING
   ├─ Fetch vectors from source
   ├─ Generate new Gemini embeddings
   ├─ Add clearance_level metadata
   └─ Upsert to target (NEW vectors)

3. PHASE B: UPDATE CHANGED
   ├─ Fetch vectors from source
   ├─ Generate new Gemini embeddings
   ├─ Preserve source metadata + clearance_level
   └─ Upsert to target (REPLACES existing)
```

### Metadata Comparison

When comparing metadata between source and target vectors:

- **Ignored keys**: `clearance_level` (added during migration, doesn't exist in source)
- **Compared keys**: All other metadata keys and values
- **Difference types**:
  - `+key` - Key exists in source but not in target
  - `-key` - Key exists in target but not in source
  - `~key` - Key exists in both but values differ

### Usage

1. Navigate to **🪞 Namespace Mirroring** in the sidebar
2. Enter the namespace ID to synchronize
3. Click **🔍 Analyze Namespace** to see what needs synchronization
4. Review the analysis results:
   - Missing vectors (to migrate)
   - Changed vectors (to update)
   - Sample metadata differences
5. Click **🪞 Start Mirroring** to execute (or use Dry Run to preview)

## Safety Features

- **Read-Only Source**: The source index (`gflops-serverless`) is never modified
- **No Deletions**: No vectors are ever deleted from either index
- **Idempotent**: Running migration/mirroring multiple times is safe
- **Dry Run Mode**: Test the migration without making any changes
- **Orphan Preservation**: Vectors only in target (not in source) are preserved

## Configuration

Configuration is loaded from environment variables (`.env` file):

```bash
# Pinecone
PINECONE_API_KEY=your-api-key
PINECONE_OLD_INDEX=gflops-serverless
PINECONE_NEW_INDEX=askdona

# Gemini
GOOGLE_AI_API_KEY=your-api-key
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_OUTPUT_DIMENSIONS=3072

# Migration
DEFAULT_NAMESPACE=your-namespace-uuid
QUERY_BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=100
UPSERT_BATCH_SIZE=100
BATCH_DELAY=0.5
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
   - Add clearance_level = 1 (integer)

5. UPSERT to target
   - Batch upsert to askdona
   - Same namespace as source
```

## API Keys

The application requires the following API keys (set in `.env` file):

| Variable | Source |
|----------|--------|
| `PINECONE_API_KEY` | [Pinecone Console](https://app.pinecone.io/) |
| `GOOGLE_AI_API_KEY` | [Google AI Studio](https://makersuite.google.com/app/apikey) |

## Files

| File | Purpose |
|------|---------|
| `app.py` | Web GUI (Streamlit) |
| `main.py` | CLI application |
| `config.py` | Configuration settings |
| `pinecone_client.py` | Pinecone API client |
| `gemini_client.py` | Gemini embedding client |
| `migration_service.py` | Core migration logic |
| `mirroring_service.py` | Namespace mirroring logic (NEW!) |
| `elasticsearch_client.py` | Elasticsearch API client |
| `requirements.txt` | Python dependencies |

## Troubleshooting

### Rate Limits

If you encounter rate limit errors:
- The application automatically retries with exponential backoff
- You can increase `BATCH_DELAY` in your `.env` file

### Missing Text

Vectors without `text` in metadata are skipped with a warning.

### Connection Issues

Ensure you have internet access and the API keys are valid.

## License

MIT License
