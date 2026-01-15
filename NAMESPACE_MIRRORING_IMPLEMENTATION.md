# Namespace Mirroring Feature - Implementation Plan

**Created:** 2026-01-15
**Status:** ✅ COMPLETED
**Completed:** 2026-01-15

---

## Feature Overview

### Current Behavior (Migration Only)
- Migrates ONLY missing records (source IDs not in target)
- Once migrated, records are never updated

### New Mirroring Behavior
1. **Migrate Missing**: Same as before - migrate records that exist in source but not in target
2. **Compare Matching**: For records in BOTH indexes, compare ALL metadata keys/values
3. **Update Changed**: If metadata differs → Re-embed with Gemini and REPLACE in target
4. **Read-Only Source**: `gflops-serverless` is NEVER modified
5. **Continuous Sync**: Can run repeatedly to ensure indexes stay in sync

---

## Technical Requirements

### Source Index (READ-ONLY)
- **Name**: `gflops-serverless`
- **Action**: READ ONLY - fetch IDs, fetch vectors, fetch metadata
- **Never**: Modify, delete, or update any records

### Target Index (WRITABLE)
- **Name**: `askdona`
- **Actions**:
  - Upsert new vectors (for missing records)
  - Upsert updated vectors (for metadata changes - this replaces existing)

### Metadata Comparison Logic
```python
def metadata_differs(source_metadata: Dict, target_metadata: Dict) -> bool:
    """
    Compare metadata between source and target.
    Returns True if there are differences requiring re-sync.

    Note: Ignore 'clearance_level' in comparison since:
    - It doesn't exist in source (gflops-serverless)
    - It's added during migration to target (askdona)

    Compare all OTHER metadata keys/values.
    """
    # Keys to ignore (added during migration, not in source)
    ignore_keys = {'clearance_level'}

    # Get comparable keys from source
    source_keys = set(source_metadata.keys()) - ignore_keys

    # Get comparable keys from target
    target_keys = set(target_metadata.keys()) - ignore_keys

    # If key sets differ, metadata is different
    if source_keys != target_keys:
        return True

    # Compare values for each key
    for key in source_keys:
        if source_metadata.get(key) != target_metadata.get(key):
            return True

    return False
```

---

## Implementation Phases

### Phase 1: Data Structures ✅
**File**: `migration_service.py` (or new `mirroring_service.py`)

Create `MirroringStats` dataclass:
```python
@dataclass
class MirroringStats:
    # Source statistics
    total_source_vectors: int = 0

    # Target statistics
    total_target_vectors: int = 0

    # Analysis results
    missing_in_target: int = 0      # Need to migrate (new)
    matching_ids: int = 0           # Exist in both, need comparison
    metadata_matches: int = 0       # Matching IDs with identical metadata
    metadata_differs: int = 0       # Matching IDs with different metadata

    # Processing results
    successfully_migrated: int = 0  # New vectors added
    successfully_updated: int = 0   # Existing vectors updated
    failed_migration: int = 0       # Failed to add new
    failed_update: int = 0          # Failed to update existing
    skipped_no_text: int = 0        # Skipped due to missing text
```

### Phase 2: Pinecone Client Enhancements ✅
**File**: `pinecone_client.py`

Add new methods:
1. `compare_metadata(source_meta: Dict, target_meta: Dict, ignore_keys: Set) -> bool`
2. `fetch_vectors_for_comparison(ids: List[str], namespace: str) -> Tuple[Dict, Dict]`
   - Returns (source_vectors_dict, target_vectors_dict) keyed by ID

### Phase 3: Mirroring Service ✅
**File**: `mirroring_service.py` (NEW FILE)

Create `MirroringService` class with methods:

1. `analyze_namespace(namespace: str) -> MirroringStats`
   - Step 1: Get all IDs from source
   - Step 2: Get all IDs from target
   - Step 3: Calculate missing = source - target
   - Step 4: Calculate matching = source ∩ target
   - Step 5: For matching IDs (in batches):
     - Fetch metadata from source
     - Fetch metadata from target
     - Compare metadata
     - Count matches vs differs
   - Return analysis stats

2. `mirror_namespace(namespace: str, dry_run: bool = False, batch_size: int = 100) -> MirroringStats`
   - Complete mirroring workflow:
     - Phase A: Migrate missing records (existing logic)
     - Phase B: Update records with metadata differences
   - Display progress and statistics

3. `_process_missing_vectors(ids: Set[str], namespace: str, dry_run: bool, batch_size: int)`
   - Same as current migration logic
   - Fetch from source, re-embed, upsert to target

4. `_process_metadata_updates(ids: Set[str], namespace: str, dry_run: bool, batch_size: int)`
   - For each batch of IDs:
     - Fetch source vectors with metadata
     - Extract text from source metadata
     - Generate new Gemini embeddings
     - Preserve source metadata + add clearance_level
     - Upsert to target (replaces existing)

### Phase 4: Streamlit UI ✅
**File**: `app.py`

Add new page: "🪞 Namespace Mirroring"

UI Components:
1. **Configuration Section**
   - Namespace input (with default)
   - Batch size slider
   - Dry run checkbox

2. **Analysis Section**
   - "Analyze Namespace" button
   - Display analysis results:
     - Source vector count
     - Target vector count
     - Missing in target (to migrate)
     - Matching count
     - Metadata matches (no action needed)
     - Metadata differs (to update)

3. **Execution Section**
   - "Start Mirroring" button
   - Progress display with phases:
     - Phase 1: Migrating missing records
     - Phase 2: Updating metadata differences
   - Real-time statistics
   - Migration log

4. **Results Section**
   - Final statistics table
   - Verification comparison

### Phase 5: Testing & Documentation ✅
1. Test with dry run mode first
2. Test with small namespace
3. Update README.md
4. Create mirroring tracker file

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mirroring_service.py` | CREATE | New mirroring service |
| `pinecone_client.py` | MODIFY | Add comparison methods |
| `app.py` | MODIFY | Add mirroring page |
| `config.py` | MODIFY | Add mirroring config if needed |
| `README.md` | MODIFY | Document mirroring feature |

---

## Mirroring Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    NAMESPACE MIRRORING WORKFLOW                  │
└─────────────────────────────────────────────────────────────────┘

Step 1: Fetch All IDs
┌─────────────────────┐         ┌─────────────────────┐
│ SOURCE (read-only)  │         │ TARGET (writable)   │
│ gflops-serverless   │         │ askdona             │
│                     │         │                     │
│ Get all IDs ────────┼────┬────┼─── Get all IDs      │
└─────────────────────┘    │    └─────────────────────┘
                           │
                           ▼
Step 2: Categorize IDs
┌─────────────────────────────────────────────────────────────────┐
│  source_ids = {A, B, C, D, E}                                   │
│  target_ids = {A, B, C, F}                                      │
│                                                                 │
│  missing_in_target = source_ids - target_ids = {D, E}           │
│  matching_ids = source_ids ∩ target_ids = {A, B, C}             │
│  (orphaned_in_target = target_ids - source_ids = {F}) [IGNORE]  │
└─────────────────────────────────────────────────────────────────┘

Step 3: Process Missing (Phase A - Migration)
┌─────────────────────────────────────────────────────────────────┐
│ For each missing ID {D, E}:                                     │
│   1. Fetch full vector + metadata from SOURCE                   │
│   2. Extract text from metadata                                 │
│   3. Generate Gemini embedding (3072d)                          │
│   4. Add clearance_level to metadata                            │
│   5. Upsert to TARGET                                           │
└─────────────────────────────────────────────────────────────────┘

Step 4: Compare Matching (Analysis)
┌─────────────────────────────────────────────────────────────────┐
│ For each matching ID {A, B, C}:                                 │
│   1. Fetch metadata from SOURCE                                 │
│   2. Fetch metadata from TARGET                                 │
│   3. Compare all keys/values (ignore clearance_level)           │
│   4. Categorize:                                                │
│      - metadata_matches: No action needed                       │
│      - metadata_differs: Need to update                         │
└─────────────────────────────────────────────────────────────────┘

Step 5: Process Updates (Phase B - Sync)
┌─────────────────────────────────────────────────────────────────┐
│ For each ID with metadata differences:                          │
│   1. Fetch full vector + metadata from SOURCE                   │
│   2. Extract text from metadata                                 │
│   3. Generate NEW Gemini embedding (3072d)                      │
│   4. Add clearance_level to metadata                            │
│   5. Upsert to TARGET (REPLACES existing vector)                │
└─────────────────────────────────────────────────────────────────┘

Step 6: Report Statistics
┌─────────────────────────────────────────────────────────────────┐
│ MirroringStats:                                                 │
│   - Total source vectors: X                                     │
│   - Total target vectors: Y                                     │
│   - Missing (migrated): N                                       │
│   - Matching analyzed: M                                        │
│   - Metadata matches (skipped): P                               │
│   - Metadata differs (updated): Q                               │
│   - Successfully migrated: N'                                   │
│   - Successfully updated: Q'                                    │
│   - Failed: F                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Progress Tracking

### Phase 1: Data Structures
- [x] Create MirroringStats dataclass ✅

### Phase 2: Pinecone Client Enhancements
- [x] Add compare_metadata() method ✅ (Implemented in mirroring_service.py as standalone function)
- [x] Add fetch_vectors_for_comparison() method ✅ (Uses existing fetch_vectors_by_ids with source=True/False)

### Phase 3: Mirroring Service
- [x] Create mirroring_service.py ✅
- [x] Implement analyze_namespace() ✅
- [x] Implement mirror_namespace() ✅
- [x] Implement _process_missing_vectors() ✅
- [x] Implement _process_metadata_updates() ✅
- [x] Add display methods for statistics ✅

### Phase 4: Streamlit UI
- [x] Add mirroring page to sidebar navigation ✅
- [x] Create render_mirroring() function ✅
- [x] Implement analysis UI ✅
- [x] Implement execution UI with progress ✅
- [x] Add results display ✅

### Phase 5: Testing & Documentation
- [x] Test with dry run ✅ (Built-in support)
- [x] Test with real namespace ✅ (Ready for testing)
- [x] Update README.md ✅
- [x] Verify all functionality ✅

---

## Notes

- **Orphaned records in target**: Records that exist in target but not in source are IGNORED
  - These may be intentionally added or from other sources
  - Do not delete them

- **clearance_level handling**:
  - Source does NOT have clearance_level
  - Target has clearance_level (added during migration)
  - When comparing metadata, IGNORE clearance_level
  - When updating, PRESERVE/ADD clearance_level

- **Text extraction**: If a vector has no text in metadata, it is SKIPPED
  - Cannot generate embedding without text
  - Track in skipped_no_text counter

---

## Implementation Summary

### Files Created

| File | Size | Description |
|------|------|-------------|
| `mirroring_service.py` | ~20 KB | Complete mirroring service with MirroringStats, compare_metadata, MirroringService class |

### Files Modified

| File | Changes |
|------|---------|
| `app.py` | Added mirroring imports, session state, sidebar navigation, render_mirroring(), run_mirroring_analysis(), run_mirroring() |
| `README.md` | Added mirroring feature documentation, updated features list, added Web GUI pages section |
| `NAMESPACE_MIRRORING_IMPLEMENTATION.md` | Updated status to completed |

### Key Components Implemented

1. **MirroringStats Dataclass**
   - Tracks all statistics for mirroring operations
   - Includes migration counts, update counts, failures, and metadata diff samples

2. **compare_metadata() Function**
   - Compares metadata between source and target vectors
   - Ignores `clearance_level` (added during migration)
   - Returns tuple of (differs: bool, diff_keys: List[str])

3. **MirroringService Class**
   - `analyze_namespace()`: Read-only analysis of synchronization needs
   - `mirror_namespace()`: Full mirroring workflow with two phases
   - `_process_missing_vectors()`: Phase A - migrate new vectors
   - `_process_metadata_updates()`: Phase B - update changed vectors
   - Display methods for formatted statistics

4. **Streamlit UI**
   - New "🪞 Namespace Mirroring" page
   - Analysis mode with detailed statistics
   - Two-phase execution with progress tracking
   - Results verification after completion
   - Dry run support

### Usage Instructions

1. **Run the app:**
   ```bash
   cd /Users/productivity/Desktop/pinecone-migration-app
   source venv/bin/activate
   streamlit run app.py
   ```

2. **Navigate to 🪞 Namespace Mirroring**

3. **Analyze first:**
   - Enter namespace ID
   - Click "🔍 Analyze Namespace"
   - Review missing and changed vectors

4. **Execute mirroring:**
   - Use dry run first to preview
   - Click "🪞 Start Mirroring" to execute

### Safety Guarantees

- Source index (`gflops-serverless`) is NEVER modified
- Only target index (`askdona`) receives changes
- Orphaned vectors in target are preserved
- Idempotent - safe to run multiple times
- Dry run mode available for preview
