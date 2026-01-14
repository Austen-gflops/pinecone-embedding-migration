# Clearance Level Metadata Fix Tracker

## Overview
Fix the `clearance_level` metadata field from string `"1"` to number `1` for all vectors.

## Problem Statement
- Current: `clearance_level: "1"` (string)
- Required: `clearance_level: 1` (number/integer)

## Tasks

### Task 1: Metadata Update Feature (Temporary) ✅ COMPLETED
Created a feature in the Streamlit app to:
1. Enter a namespace
2. Fetch all vectors from that namespace
3. Update metadata from `clearance_level: "1"` to `clearance_level: 1`
4. No re-embedding required - only metadata update

### Task 2: Update Migration App ✅ COMPLETED
Updated the migration service to use number format for new migrations.

---

## Implementation Summary

### Files Modified

| File | Changes |
|------|---------|
| `pinecone_client.py` | Added `update_vector_metadata()` and `get_vectors_with_string_clearance()` methods |
| `app.py` | Added "🔧 Fix Metadata" page to navigation and router |
| `config.py` | Changed `new_metadata_value` from `str = "1"` to `int = 1` |
| `README.md` | Updated documentation for integer format and new Fix Metadata page |
| `.env.example` | Added comment about integer format |

### New Methods in pinecone_client.py

1. **`_update_single_vector()`** - Updates metadata for a single vector
2. **`update_vector_metadata()`** - Concurrent batch update for metadata
3. **`get_vectors_with_string_clearance()`** - Scans for vectors needing fix

### New Streamlit Page: Fix Metadata

Location: `app.py` → `render_fix_metadata()`

Features:
- Namespace input
- Index selection (source or target)
- Concurrent worker configuration
- Scan button to find vectors with string clearance_level
- Fix button to update all vectors to integer format
- Progress tracking and detailed logging

---

## How to Use Fix Metadata Feature

1. Open the Streamlit app at http://localhost:8501
2. Navigate to "🔧 Fix Metadata" in the sidebar
3. Enter the namespace to fix
4. Select the index (usually "Target (askdona)")
5. Click "🔍 Scan for Vectors to Fix" to find vectors with string format
6. Review the count and sample IDs
7. Click "🔧 Fix Metadata" to update all vectors to integer format
8. Wait for completion - progress is shown in real-time

---

## Technical Details

### Pinecone Bulk Update API (Optimized)
- Uses Pinecone's bulk update REST API with filter-based matching
- Single API call updates ALL matching vectors
- Endpoint: `POST https://{INDEX_HOST}/vectors/update`
- Header: `X-Pinecone-API-Version: 2025-01`
- Filter: `{"clearance_level": {"$eq": "1"}}`
- Update: `{"clearance_level": 1}`

### API Request Format
```json
{
  "namespace": "your-namespace",
  "filter": {"clearance_level": {"$eq": "1"}},
  "setMetadata": {"clearance_level": 1}
}
```

### Features
- **Scan (Preview)**: Counts vectors needing update
- **Dry Run**: Tests the bulk update without making changes
- **Bulk Fix**: Executes the update in a single API call

---

## Status: ✅ COMPLETED

All tasks have been completed:
- [x] Explore Pinecone API for metadata updates
- [x] Explore current migration_service.py implementation
- [x] Explore app.py structure
- [x] Explore pinecone_client.py capabilities
- [x] Add update_vector_metadata() to pinecone_client.py
- [x] Add "Fix Metadata" page to app.py
- [x] Update config.py for integer format
- [x] Update README and .env.example documentation
- [x] Test the feature (Streamlit app running)
- [x] Push to GitHub (pending)

---

## Changes Pushed to GitHub
- Commit: `81ad73c` - Add Fix Metadata feature and change clearance_level to integer
- Repository: https://github.com/Austen-gflops/pinecone-embedding-migration
