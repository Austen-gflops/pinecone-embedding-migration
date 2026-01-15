"""
Pinecone Vector Embedding Migration Tool - Web GUI

A Streamlit-based web interface for migrating vector embeddings
from one Pinecone index to another using Gemini embeddings.
"""
import streamlit as st
import time
from typing import Dict, List, Any, Optional

# Must be the first Streamlit command
st.set_page_config(
    page_title="Pinecone Migration Tool",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
from config import config
from pinecone_client import pinecone_client, VectorRecord
from gemini_client import GeminiEmbeddingClient
from elasticsearch_client import elasticsearch_client
from mirroring_service import mirroring_service, MirroringStats, compare_metadata, METADATA_IGNORE_KEYS


def init_session_state():
    """Initialize session state variables"""
    if 'migration_running' not in st.session_state:
        st.session_state.migration_running = False
    if 'migration_stats' not in st.session_state:
        st.session_state.migration_stats = None
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None
    if 'mirroring_stats' not in st.session_state:
        st.session_state.mirroring_stats = None
    if 'mirroring_analysis_done' not in st.session_state:
        st.session_state.mirroring_analysis_done = False


def get_gemini_client():
    """Get or create Gemini client"""
    if st.session_state.gemini_client is None:
        st.session_state.gemini_client = GeminiEmbeddingClient()
    return st.session_state.gemini_client


def render_sidebar():
    """Render the sidebar with navigation and info"""
    with st.sidebar:
        st.title("🔄 Migration Tool")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Check Namespace", "🪞 Namespace Mirroring", "🚀 Pinecone ReEmbedding", "🔎 Elasticsearch Update", "⚙️ Configuration"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats
        st.subheader("Quick Info")
        st.markdown(f"**Source:** `{config.pinecone.old_index_name}`")
        st.markdown(f"**Target:** `{config.pinecone.new_index_name}`")
        st.markdown(f"**Model:** `{config.gemini.model}`")
        st.markdown(f"**Dimensions:** `{config.gemini.output_dimensions}`")

        st.markdown("---")

        # New metadata info
        st.subheader("New Metadata")
        st.code(f'{config.migration.new_metadata_key}: {config.migration.new_metadata_value}')  # Integer, no quotes

        return page


def render_dashboard():
    """Render the main dashboard"""
    st.title("📊 Migration Dashboard")
    st.markdown("Overview of your Pinecone indexes and migration status.")

    col1, col2 = st.columns(2)

    # Source index stats
    with col1:
        st.subheader(f"🗄️ Source: {config.pinecone.old_index_name}")
        with st.spinner("Fetching source stats..."):
            source_stats = pinecone_client.get_namespace_stats(
                config.pinecone.old_index_name,
                config.migration.default_namespace
            )

        if source_stats.get("exists"):
            st.metric("Vector Count", f"{source_stats.get('vector_count', 0):,}")
            st.success("Namespace exists")
        else:
            st.metric("Vector Count", "0")
            st.warning("Namespace not found")

    # Target index stats
    with col2:
        st.subheader(f"🎯 Target: {config.pinecone.new_index_name}")
        with st.spinner("Fetching target stats..."):
            target_stats = pinecone_client.get_namespace_stats(
                config.pinecone.new_index_name,
                config.migration.default_namespace
            )

        if target_stats.get("exists"):
            st.metric("Vector Count", f"{target_stats.get('vector_count', 0):,}")
            st.success("Namespace exists")
        else:
            st.metric("Vector Count", "0")
            st.info("Namespace will be created on migration")

    # Migration summary
    st.markdown("---")
    st.subheader("📈 Migration Summary")

    source_count = source_stats.get('vector_count', 0)
    target_count = target_stats.get('vector_count', 0)
    remaining = source_count - target_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Source Vectors", f"{source_count:,}")
    col2.metric("Already Migrated", f"{target_count:,}")
    col3.metric("Remaining", f"{remaining:,}", delta=f"-{remaining:,}" if remaining > 0 else None)

    if remaining > 0:
        progress = target_count / source_count if source_count > 0 else 0
        st.progress(progress, text=f"Migration Progress: {progress:.1%}")
    elif source_count > 0 and remaining == 0:
        st.success("🎉 Migration Complete! All vectors have been migrated.")

    # Default namespace display
    st.markdown("---")
    st.subheader("🏷️ Default Namespace")
    st.code(config.migration.default_namespace)

    # Elasticsearch section
    st.markdown("---")
    st.title("🔎 Elasticsearch Status")
    st.markdown("Overview of Elasticsearch keyword search index.")

    # Connection test
    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Testing Elasticsearch connection..."):
            es_connected = elasticsearch_client.test_connection()

        if es_connected:
            st.success("✅ Connected to Elasticsearch")
        else:
            st.error("❌ Elasticsearch connection failed")
            return

    with col2:
        es_index_stats = elasticsearch_client.get_index_stats()
        st.metric("Index", config.elasticsearch.index_name)
        st.metric("Total Documents", f"{es_index_stats.get('doc_count', 0):,}")

    # Elasticsearch stats
    st.markdown("---")
    st.subheader("📈 Elasticsearch Update Progress")

    with st.spinner("Fetching Elasticsearch statistics..."):
        es_all_stats = elasticsearch_client.get_all_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", f"{es_all_stats.total_count:,}")
    col2.metric("With clearance_level", f"{es_all_stats.with_clearance_level:,}")
    col3.metric("Without clearance_level", f"{es_all_stats.without_clearance_level:,}")

    if es_all_stats.total_count > 0:
        es_progress = es_all_stats.with_clearance_level / es_all_stats.total_count
        st.progress(es_progress, text=f"Update Progress: {es_progress:.1%}")

        if es_all_stats.without_clearance_level == 0:
            st.success("🎉 All Elasticsearch documents have clearance_level!")
        else:
            st.info(f"📝 {es_all_stats.without_clearance_level:,} documents still need clearance_level added")


def render_check_namespace():
    """Render the namespace checker page"""
    st.title("🔍 Check Namespace")
    st.markdown("Check the embedding count for any namespace in either index.")

    # Input
    namespace = st.text_input(
        "Namespace ID",
        value=config.migration.default_namespace,
        help="Enter the namespace UUID to check"
    )

    index_choice = st.selectbox(
        "Select Index",
        ["Both", "Source (gflops-serverless)", "Target (askdona)"]
    )

    if st.button("🔍 Check Namespace", type="primary"):
        if not namespace:
            st.error("Please enter a namespace ID")
            return

        col1, col2 = st.columns(2)

        if index_choice in ["Both", "Source (gflops-serverless)"]:
            with col1:
                st.subheader(f"Source: {config.pinecone.old_index_name}")
                with st.spinner("Checking..."):
                    stats = pinecone_client.get_namespace_stats(
                        config.pinecone.old_index_name, namespace
                    )

                st.metric("Vector Count", f"{stats.get('vector_count', 0):,}")
                if stats.get("exists"):
                    st.success("✅ Namespace exists")
                else:
                    st.warning("⚠️ Namespace not found")

        if index_choice in ["Both", "Target (askdona)"]:
            with col2:
                st.subheader(f"Target: {config.pinecone.new_index_name}")
                with st.spinner("Checking..."):
                    stats = pinecone_client.get_namespace_stats(
                        config.pinecone.new_index_name, namespace
                    )

                st.metric("Vector Count", f"{stats.get('vector_count', 0):,}")
                if stats.get("exists"):
                    st.success("✅ Namespace exists")
                else:
                    st.info("ℹ️ Namespace will be created on migration")


def render_mirroring():
    """Render the namespace mirroring page"""
    st.title("🪞 Namespace Mirroring")
    st.markdown("Continuously synchronize vectors between source and target indexes.")

    # Warning/Info
    st.info("""
    **Namespace Mirroring** ensures your target index stays synchronized with the source:

    1. **Migrate Missing** - Vectors in source but not in target are migrated
    2. **Update Changed** - Vectors with metadata differences are re-embedded and updated

    ⚠️ **Source index is READ-ONLY** - Only the target index is ever modified.
    """)

    # Configuration
    st.subheader("⚙️ Mirroring Settings")

    col1, col2 = st.columns(2)

    with col1:
        namespace = st.text_input(
            "Namespace to Mirror",
            value=config.migration.default_namespace,
            help="The namespace to synchronize between indexes"
        )

        batch_size = st.number_input(
            "Batch Size",
            min_value=10,
            max_value=100,
            value=100,
            help="Number of vectors to process at a time"
        )

    with col2:
        dry_run = st.checkbox(
            "🧪 Dry Run",
            value=False,
            help="Preview changes without making modifications"
        )

        st.markdown(f"**Source:** `{config.pinecone.old_index_name}` (read-only)")
        st.markdown(f"**Target:** `{config.pinecone.new_index_name}` (writable)")

    # Metadata info
    st.markdown("---")
    st.subheader("🏷️ Metadata Handling")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Added during sync:**")
        st.code(f'{config.migration.new_metadata_key}: {config.migration.new_metadata_value}')

    with col2:
        st.markdown("**Ignored in comparison:**")
        st.code(', '.join(METADATA_IGNORE_KEYS))

    # Analysis Section
    st.markdown("---")
    st.subheader("📋 Namespace Analysis")
    st.markdown("Analyze the namespace to see what needs synchronization.")

    if st.button("🔍 Analyze Namespace", type="secondary"):
        run_mirroring_analysis(namespace)

    # Show analysis results if available
    if st.session_state.mirroring_stats and st.session_state.mirroring_analysis_done:
        stats = st.session_state.mirroring_stats

        st.markdown("#### Analysis Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Source Vectors", f"{stats.total_source_vectors:,}")
        col2.metric("Target Vectors", f"{stats.total_target_vectors:,}")
        col3.metric("Missing (to migrate)", f"{stats.missing_in_target:,}")
        col4.metric("Different (to update)", f"{stats.metadata_differs:,}")

        # Detailed breakdown
        if stats.matching_ids > 0:
            st.markdown("#### Matching Vectors Analysis")
            col1, col2 = st.columns(2)
            col1.metric("Matching IDs", f"{stats.matching_ids:,}")
            col2.metric("Metadata Matches", f"{stats.metadata_matches:,}")

        # Show sample differences if any
        if stats.metadata_diff_details:
            with st.expander("📝 Sample Metadata Differences", expanded=False):
                for i, detail in enumerate(stats.metadata_diff_details[:5], 1):
                    st.markdown(f"**{i}. Vector ID:** `{detail['id'][:30]}...`")
                    st.markdown(f"   Changed keys: `{', '.join(detail['diff_keys'])}`")

        # Action summary
        total_actions = stats.missing_in_target + stats.metadata_differs
        if total_actions > 0:
            st.warning(f"""
            **Actions Required:**
            - {stats.missing_in_target:,} vectors to migrate (new)
            - {stats.metadata_differs:,} vectors to update (changed)
            - **Total: {total_actions:,} vectors**
            """)
        else:
            st.success("✅ Namespace is fully synchronized! No actions needed.")

    # Execute Mirroring Section
    st.markdown("---")
    st.subheader("▶️ Execute Mirroring")

    if st.session_state.mirroring_stats:
        stats = st.session_state.mirroring_stats
        total_actions = stats.missing_in_target + stats.metadata_differs

        if total_actions > 0:
            st.warning(f"""
            **Ready to mirror:**
            - Namespace: `{namespace}`
            - To migrate: {stats.missing_in_target:,} vectors
            - To update: {stats.metadata_differs:,} vectors
            - Mode: {'DRY RUN' if dry_run else 'LIVE'}
            """)

            if st.button("🪞 Start Mirroring" + (" (Dry Run)" if dry_run else ""), type="primary"):
                run_mirroring(namespace, batch_size, dry_run, skip_analysis=True)
        else:
            st.success("No mirroring needed - namespace is already synchronized!")
    else:
        st.info("👆 Run analysis first to see what needs synchronization.")

        if st.button("🪞 Start Full Mirroring" + (" (Dry Run)" if dry_run else ""), type="primary"):
            run_mirroring(namespace, batch_size, dry_run, skip_analysis=False)


def run_mirroring_analysis(namespace: str):
    """Run the namespace analysis for mirroring"""

    # Status containers
    status_container = st.empty()
    progress_container = st.empty()

    status_container.info("🔍 Analyzing namespace...")

    try:
        # Step 1: Get source IDs
        status_container.info("Step 1/4: Fetching source vector IDs...")
        source_ids = pinecone_client.get_all_vector_ids(namespace, source=True)

        # Step 2: Get target IDs
        status_container.info("Step 2/4: Fetching target vector IDs...")
        target_ids = pinecone_client.get_all_vector_ids(namespace, source=False)

        # Step 3: Categorize
        status_container.info("Step 3/4: Categorizing vectors...")
        missing_in_target = source_ids - target_ids
        matching_ids = source_ids & target_ids

        # Initialize stats
        stats = MirroringStats()
        stats.total_source_vectors = len(source_ids)
        stats.total_target_vectors = len(target_ids)
        stats.missing_in_target = len(missing_in_target)
        stats.matching_ids = len(matching_ids)

        # Step 4: Compare metadata for matching IDs
        if matching_ids:
            status_container.info(f"Step 4/4: Comparing metadata for {len(matching_ids):,} matching vectors...")

            progress_bar = progress_container.progress(0)
            matching_list = list(matching_ids)
            batch_size = 100
            total_batches = (len(matching_list) + batch_size - 1) // batch_size

            samples_collected = 0
            max_samples = 5

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(matching_list))
                batch_ids = matching_list[start_idx:end_idx]

                # Fetch from both indexes
                source_vectors = pinecone_client.fetch_vectors_by_ids(
                    batch_ids, namespace, source=True, show_progress=False
                )
                target_vectors = pinecone_client.fetch_vectors_by_ids(
                    batch_ids, namespace, source=False, show_progress=False
                )

                # Create lookup dicts
                source_dict = {v.id: v for v in source_vectors}
                target_dict = {v.id: v for v in target_vectors}

                # Compare each pair
                for vec_id in batch_ids:
                    source_vec = source_dict.get(vec_id)
                    target_vec = target_dict.get(vec_id)

                    if source_vec and target_vec:
                        differs, diff_keys = compare_metadata(
                            source_vec.metadata,
                            target_vec.metadata
                        )

                        if differs:
                            stats.metadata_differs += 1
                            if samples_collected < max_samples:
                                stats.metadata_diff_details.append({
                                    'id': vec_id,
                                    'diff_keys': diff_keys
                                })
                                samples_collected += 1
                        else:
                            stats.metadata_matches += 1

                # Update progress
                progress = (batch_idx + 1) / total_batches
                progress_bar.progress(progress)

            progress_container.empty()
        else:
            status_container.info("Step 4/4: No matching vectors to compare")

        # Save to session state
        st.session_state.mirroring_stats = stats
        st.session_state.mirroring_analysis_done = True

        status_container.success("✅ Analysis complete!")

    except Exception as e:
        status_container.error(f"❌ Analysis failed: {e}")
        st.exception(e)


def run_mirroring(namespace: str, batch_size: int, dry_run: bool, skip_analysis: bool = False):
    """Execute the mirroring process"""

    # Get clients
    gemini = get_gemini_client()

    # Status containers
    status_container = st.empty()
    progress_bar = st.progress(0)
    stats_container = st.empty()
    log_container = st.expander("📜 Mirroring Log", expanded=True)

    with log_container:
        st.write("🪞 Starting namespace mirroring...")
        if dry_run:
            st.write("⚠️ DRY RUN MODE - No changes will be made")

    try:
        # Use existing analysis if available
        if skip_analysis and st.session_state.mirroring_stats:
            stats = st.session_state.mirroring_stats
            with log_container:
                st.write(f"📊 Using existing analysis: {stats.missing_in_target} to migrate, {stats.metadata_differs} to update")
        else:
            # Run fresh analysis
            with log_container:
                st.write("📊 Running fresh analysis...")

            status_container.info("Analyzing namespace...")

            source_ids = pinecone_client.get_all_vector_ids(namespace, source=True)
            target_ids = pinecone_client.get_all_vector_ids(namespace, source=False)

            stats = MirroringStats()
            stats.total_source_vectors = len(source_ids)
            stats.total_target_vectors = len(target_ids)
            stats.missing_in_target = len(source_ids - target_ids)
            stats.matching_ids = len(source_ids & target_ids)

            # Compare metadata for matching IDs
            matching_ids = list(source_ids & target_ids)
            if matching_ids:
                for i in range(0, len(matching_ids), 100):
                    batch_ids = matching_ids[i:i + 100]
                    source_vecs = pinecone_client.fetch_vectors_by_ids(batch_ids, namespace, source=True, show_progress=False)
                    target_vecs = pinecone_client.fetch_vectors_by_ids(batch_ids, namespace, source=False, show_progress=False)

                    source_dict = {v.id: v for v in source_vecs}
                    target_dict = {v.id: v for v in target_vecs}

                    for vid in batch_ids:
                        if vid in source_dict and vid in target_dict:
                            differs, _ = compare_metadata(source_dict[vid].metadata, target_dict[vid].metadata)
                            if differs:
                                stats.metadata_differs += 1
                            else:
                                stats.metadata_matches += 1

            st.session_state.mirroring_stats = stats

        # Check if any work needed
        total_work = stats.missing_in_target + stats.metadata_differs
        if total_work == 0:
            status_container.success("🎉 Namespace is already synchronized!")
            progress_bar.progress(1.0)
            return

        with log_container:
            st.write(f"📋 Work to do: {stats.missing_in_target} to migrate, {stats.metadata_differs} to update")

        # ============ PHASE A: Migrate Missing Vectors ============
        if stats.missing_in_target > 0:
            status_container.info(f"Phase A: Migrating {stats.missing_in_target} missing vectors...")

            with log_container:
                st.write(f"🔄 Phase A: Migrating {stats.missing_in_target} missing vectors...")

            # Get missing IDs
            source_ids = pinecone_client.get_all_vector_ids(namespace, source=True)
            target_ids = pinecone_client.get_all_vector_ids(namespace, source=False)
            missing_ids = list(source_ids - target_ids)

            # Fetch vectors
            vectors_to_migrate = pinecone_client.fetch_vectors_by_ids(missing_ids, namespace, source=True)

            total_batches = (len(vectors_to_migrate) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(vectors_to_migrate))
                batch = vectors_to_migrate[start_idx:end_idx]

                # Extract texts
                texts = []
                valid_vectors = []
                for vec in batch:
                    text = vec.metadata.get("text", "")
                    if text and text.strip():
                        texts.append(text)
                        valid_vectors.append(vec)
                    else:
                        stats.skipped_no_text += 1

                if not texts:
                    continue

                # Generate embeddings
                try:
                    embeddings = gemini._generate_batch_embeddings(texts)

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
                            stats.failed_migration += 1

                    # Upsert
                    if vectors_to_upsert and not dry_run:
                        pinecone_client.upsert_vectors(vectors_to_upsert, namespace)
                        stats.successfully_migrated += len(vectors_to_upsert)
                    elif vectors_to_upsert:
                        stats.successfully_migrated += len(vectors_to_upsert)

                except Exception as e:
                    stats.failed_migration += len(valid_vectors)
                    with log_container:
                        st.write(f"❌ Error in migration batch: {e}")

                # Update progress (Phase A is 50% of total)
                phase_a_progress = (batch_idx + 1) / total_batches * 0.5
                progress_bar.progress(phase_a_progress)

                stats_container.markdown(f"""
                **Phase A Progress:** Batch {batch_idx + 1}/{total_batches} |
                **Migrated:** {stats.successfully_migrated:,} |
                **Failed:** {stats.failed_migration}
                """)

                time.sleep(0.3)

            with log_container:
                st.write(f"✅ Phase A complete: {stats.successfully_migrated} migrated")

        # ============ PHASE B: Update Changed Vectors ============
        if stats.metadata_differs > 0:
            status_container.info(f"Phase B: Updating {stats.metadata_differs} changed vectors...")

            with log_container:
                st.write(f"🔄 Phase B: Updating vectors with metadata differences...")

            # Find IDs with differences
            source_ids = pinecone_client.get_all_vector_ids(namespace, source=True)
            target_ids = pinecone_client.get_all_vector_ids(namespace, source=False)
            matching_ids = list(source_ids & target_ids)

            ids_to_update = []
            for i in range(0, len(matching_ids), 100):
                batch_ids = matching_ids[i:i + 100]
                source_vecs = pinecone_client.fetch_vectors_by_ids(batch_ids, namespace, source=True, show_progress=False)
                target_vecs = pinecone_client.fetch_vectors_by_ids(batch_ids, namespace, source=False, show_progress=False)

                source_dict = {v.id: v for v in source_vecs}
                target_dict = {v.id: v for v in target_vecs}

                for vid in batch_ids:
                    if vid in source_dict and vid in target_dict:
                        differs, _ = compare_metadata(source_dict[vid].metadata, target_dict[vid].metadata)
                        if differs:
                            ids_to_update.append(vid)

            if ids_to_update:
                # Fetch source vectors for update
                vectors_to_update = pinecone_client.fetch_vectors_by_ids(ids_to_update, namespace, source=True)
                total_batches = (len(vectors_to_update) + batch_size - 1) // batch_size

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(vectors_to_update))
                    batch = vectors_to_update[start_idx:end_idx]

                    # Extract texts
                    texts = []
                    valid_vectors = []
                    for vec in batch:
                        text = vec.metadata.get("text", "")
                        if text and text.strip():
                            texts.append(text)
                            valid_vectors.append(vec)
                        else:
                            stats.skipped_no_text += 1

                    if not texts:
                        continue

                    # Generate embeddings
                    try:
                        embeddings = gemini._generate_batch_embeddings(texts)

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
                                stats.failed_update += 1

                        # Upsert (replaces existing)
                        if vectors_to_upsert and not dry_run:
                            pinecone_client.upsert_vectors(vectors_to_upsert, namespace)
                            stats.successfully_updated += len(vectors_to_upsert)
                        elif vectors_to_upsert:
                            stats.successfully_updated += len(vectors_to_upsert)

                    except Exception as e:
                        stats.failed_update += len(valid_vectors)
                        with log_container:
                            st.write(f"❌ Error in update batch: {e}")

                    # Update progress (Phase B is remaining 50%)
                    phase_b_progress = 0.5 + (batch_idx + 1) / total_batches * 0.5
                    progress_bar.progress(phase_b_progress)

                    stats_container.markdown(f"""
                    **Phase B Progress:** Batch {batch_idx + 1}/{total_batches} |
                    **Updated:** {stats.successfully_updated:,} |
                    **Failed:** {stats.failed_update}
                    """)

                    time.sleep(0.3)

            with log_container:
                st.write(f"✅ Phase B complete: {stats.successfully_updated} updated")

        # Final status
        progress_bar.progress(1.0)

        total_processed = stats.successfully_migrated + stats.successfully_updated
        total_failed = stats.failed_migration + stats.failed_update

        if dry_run:
            status_container.success(f"🧪 Dry Run Complete! Would process {total_processed:,} vectors")
        else:
            if total_failed == 0:
                status_container.success(f"🎉 Mirroring Complete! Processed {total_processed:,} vectors")
            else:
                status_container.warning(f"⚠️ Mirroring completed with {total_failed} failures")

        # Final stats display
        st.markdown("---")
        st.subheader("📊 Mirroring Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Migrated (new)", f"{stats.successfully_migrated:,}")
        col2.metric("Updated (changed)", f"{stats.successfully_updated:,}")
        col3.metric("Failed", f"{total_failed}")
        col4.metric("Skipped (no text)", f"{stats.skipped_no_text}")

        # Verification
        if not dry_run:
            st.markdown("---")
            st.subheader("✅ Verification")

            with st.spinner("Verifying synchronization..."):
                new_source_count = len(pinecone_client.get_all_vector_ids(namespace, source=True))
                new_target_count = len(pinecone_client.get_all_vector_ids(namespace, source=False))

            col1, col2, col3 = st.columns(3)
            col1.metric("Source Vectors", f"{new_source_count:,}")
            col2.metric("Target Vectors", f"{new_target_count:,}")
            col3.metric("Difference", f"{new_source_count - new_target_count:,}")

            if new_source_count == new_target_count:
                st.success("✅ Vector counts match! Namespace is synchronized.")
            else:
                st.info(f"ℹ️ Difference of {new_source_count - new_target_count} vectors (may be skipped due to no text)")

        # Clear analysis state
        st.session_state.mirroring_analysis_done = False

    except Exception as e:
        status_container.error(f"❌ Mirroring failed: {e}")
        st.exception(e)


def render_migration():
    """Render the Pinecone re-embedding page"""
    st.title("🚀 Pinecone ReEmbedding")
    st.markdown("Migrate vector embeddings from source to target index.")

    # Warning
    st.warning("""
    **Important Notes:**
    - Source index is READ-ONLY (no deletions)
    - All original metadata is preserved
    - New metadata `clearance_level: "1"` is added to all vectors
    - Already migrated vectors are automatically skipped
    """)

    # Configuration
    st.subheader("⚙️ Migration Settings")

    col1, col2 = st.columns(2)

    with col1:
        namespace = st.text_input(
            "Namespace to Migrate",
            value=config.migration.default_namespace,
            help="The namespace to migrate from source to target"
        )

        batch_size = st.number_input(
            "Batch Size",
            min_value=10,
            max_value=100,
            value=100,
            help="Number of vectors to process at a time (optimized: 100)"
        )

    with col2:
        dry_run = st.checkbox(
            "🧪 Dry Run",
            value=False,
            help="Preview migration without making changes"
        )

        st.markdown(f"**Embedding Model:** `{config.gemini.model}`")
        st.markdown(f"**Dimensions:** `{config.gemini.output_dimensions}`")

    # Pre-migration check
    st.markdown("---")
    st.subheader("📋 Pre-Migration Check")

    if st.button("🔍 Check Migration Status"):
        with st.spinner("Checking..."):
            source_stats = pinecone_client.get_namespace_stats(
                config.pinecone.old_index_name, namespace
            )
            target_stats = pinecone_client.get_namespace_stats(
                config.pinecone.new_index_name, namespace
            )

        source_count = source_stats.get('vector_count', 0)
        target_count = target_stats.get('vector_count', 0)
        remaining = source_count - target_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Source Vectors", f"{source_count:,}")
        col2.metric("Already Migrated", f"{target_count:,}")
        col3.metric("To Migrate", f"{remaining:,}")

        if remaining == 0 and source_count > 0:
            st.success("🎉 All vectors already migrated!")
        elif remaining > 0:
            st.info(f"Ready to migrate {remaining:,} vectors")

    # Run migration
    st.markdown("---")
    st.subheader("▶️ Execute Migration")

    if st.button("🚀 Start Migration" + (" (Dry Run)" if dry_run else ""), type="primary"):
        run_migration(namespace, batch_size, dry_run)


def run_migration(namespace: str, batch_size: int, dry_run: bool):
    """Execute the migration process"""

    # Get clients
    gemini = get_gemini_client()

    # Status containers
    status_container = st.empty()
    progress_bar = st.progress(0)
    stats_container = st.empty()
    log_container = st.expander("📜 Migration Log", expanded=True)

    with log_container:
        st.write("🔄 Starting migration...")

    try:
        # Step 1: Get existing target IDs
        status_container.info("Step 1/5: Fetching existing target IDs...")
        with log_container:
            st.write("Fetching IDs from target index...")

        target_ids = pinecone_client.get_all_vector_ids(namespace, source=False)
        already_migrated = len(target_ids)

        with log_container:
            st.write(f"✅ Found {already_migrated:,} vectors already in target")

        # Step 2: Get source IDs
        status_container.info("Step 2/5: Fetching source vector IDs...")
        with log_container:
            st.write("Fetching IDs from source index...")

        source_ids = pinecone_client.get_all_vector_ids(namespace, source=True)
        total_source = len(source_ids)

        with log_container:
            st.write(f"✅ Found {total_source:,} vectors in source")

        # Step 3: Calculate what needs migration
        ids_to_migrate = source_ids - target_ids
        to_migrate_count = len(ids_to_migrate)

        with log_container:
            st.write(f"📊 Vectors to migrate: {to_migrate_count:,}")

        if to_migrate_count == 0:
            status_container.success("🎉 All vectors already migrated!")
            progress_bar.progress(1.0)
            return

        # Step 4: Fetch source vectors
        status_container.info("Step 3/5: Fetching source vector metadata...")
        with log_container:
            st.write("Fetching vector metadata from source...")

        vectors_to_migrate = pinecone_client.fetch_vectors_by_ids(
            list(ids_to_migrate), namespace, source=True
        )

        with log_container:
            st.write(f"✅ Fetched {len(vectors_to_migrate):,} vectors with metadata")

        # Step 5: Process in batches
        status_container.info("Step 4/5: Generating embeddings and upserting...")

        total_batches = (len(vectors_to_migrate) + batch_size - 1) // batch_size
        successfully_migrated = 0
        failed = 0
        skipped_no_text = 0

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(vectors_to_migrate))
            batch = vectors_to_migrate[start_idx:end_idx]

            # Extract texts
            texts = []
            valid_vectors = []
            for vec in batch:
                text = vec.metadata.get("text", "")
                if text and text.strip():
                    texts.append(text)
                    valid_vectors.append(vec)
                else:
                    skipped_no_text += 1

            if not texts:
                progress_bar.progress((batch_idx + 1) / total_batches)
                continue

            # Generate embeddings
            try:
                embeddings = gemini._generate_batch_embeddings(texts)

                # Prepare vectors for upsert
                vectors_to_upsert = []
                for vec, embedding in zip(valid_vectors, embeddings):
                    new_metadata = dict(vec.metadata)
                    new_metadata[config.migration.new_metadata_key] = config.migration.new_metadata_value

                    vectors_to_upsert.append({
                        "id": vec.id,
                        "values": embedding,
                        "metadata": new_metadata
                    })

                # Upsert to target
                if not dry_run:
                    pinecone_client.upsert_vectors(vectors_to_upsert, namespace)

                successfully_migrated += len(vectors_to_upsert)

            except Exception as e:
                failed += len(valid_vectors)
                with log_container:
                    st.write(f"❌ Error in batch {batch_idx + 1}: {e}")

            # Update progress
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress)

            stats_container.markdown(f"""
            **Progress:** Batch {batch_idx + 1}/{total_batches} |
            **Migrated:** {successfully_migrated:,} |
            **Failed:** {failed} |
            **Skipped (no text):** {skipped_no_text}
            """)

            # Small delay between batches
            time.sleep(0.5)

        # Final status
        progress_bar.progress(1.0)

        if dry_run:
            status_container.success(f"🧪 Dry Run Complete! Would have migrated {successfully_migrated:,} vectors")
        else:
            status_container.success(f"🎉 Migration Complete! Migrated {successfully_migrated:,} vectors")

        # Final stats
        st.markdown("---")
        st.subheader("📊 Migration Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Successfully Migrated", f"{successfully_migrated:,}")
        col2.metric("Failed", f"{failed}")
        col3.metric("Skipped (no text)", f"{skipped_no_text}")
        col4.metric("Already Migrated", f"{already_migrated:,}")

    except Exception as e:
        status_container.error(f"❌ Migration failed: {e}")
        st.exception(e)


def render_elasticsearch_update():
    """Render the Elasticsearch metadata update page"""
    st.title("🔎 Elasticsearch Update")
    st.markdown("Add `clearance_level: 1` (integer) to Elasticsearch documents.")

    # Warning
    st.warning("""
    **Important Notes:**
    - **NO DELETIONS** - This only ADDS the new metadata field
    - **PRESERVES ALL DATA** - Existing fields are not modified
    - **INTEGER VALUE** - clearance_level is stored as `1` (not "1")
    - Filter by namespace using the `pinecone_namespace` field
    """)

    # Check connection first
    with st.spinner("Checking Elasticsearch connection..."):
        connected = elasticsearch_client.test_connection()

    if not connected:
        st.error("❌ Elasticsearch connection failed. Check the Dashboard for details.")
        return

    # Namespace selection
    st.subheader("🏷️ Update by Namespace")

    # Get available namespaces
    with st.spinner("Fetching namespaces..."):
        namespaces = elasticsearch_client.list_all_namespaces()

    if not namespaces:
        st.warning("No namespaces found in the index.")
        return

    namespace_options = ["All Namespaces"] + namespaces

    # Set default to config.migration.default_namespace if it exists in the list
    default_index = 0  # "All Namespaces"
    if config.migration.default_namespace in namespaces:
        default_index = namespace_options.index(config.migration.default_namespace)

    selected_namespace = st.selectbox(
        "Select Namespace",
        namespace_options,
        index=default_index,
        help="Select a namespace to update, or 'All Namespaces' to update everything"
    )

    namespace_filter = None if selected_namespace == "All Namespaces" else selected_namespace

    # Show namespace stats
    if namespace_filter:
        with st.spinner("Fetching namespace statistics..."):
            ns_stats = elasticsearch_client.get_namespace_stats(namespace_filter)

        col1, col2, col3 = st.columns(3)
        col1.metric("Documents in Namespace", f"{ns_stats.total_count:,}")
        col2.metric("With clearance_level", f"{ns_stats.with_clearance_level:,}")
        col3.metric("Without clearance_level", f"{ns_stats.without_clearance_level:,}")

    # Update options
    st.markdown("---")
    st.subheader("⚙️ Update Options")

    only_missing = st.checkbox(
        "Only update documents without clearance_level",
        value=True,
        help="If checked, documents that already have clearance_level will be skipped"
    )

    # Preview
    st.markdown("---")
    st.subheader("👁️ Preview")

    if st.button("🔍 Preview Update"):
        with st.spinner("Counting documents to update..."):
            preview_count = elasticsearch_client.preview_update(namespace_filter, only_missing)

        st.info(f"**{preview_count:,}** documents would be updated")

        if preview_count > 0:
            st.markdown("**Sample documents:**")
            samples = elasticsearch_client.sample_documents(namespace_filter, 5)
            for i, doc in enumerate(samples, 1):
                with st.expander(f"Document {i}: {doc.get('filename', 'Unknown')}"):
                    st.json(doc)

    # Execute update
    st.markdown("---")
    st.subheader("▶️ Execute Update")

    st.warning(f"""
    **Ready to update:**
    - Namespace: `{selected_namespace}`
    - Only missing: `{only_missing}`
    - New field: `clearance_level: {config.migration.new_metadata_value}` (integer)
    """)

    if st.button("🚀 Start Elasticsearch Update", type="primary"):
        run_elasticsearch_update(namespace_filter, only_missing)


def run_elasticsearch_update(namespace: str, only_missing: bool):
    """Execute the Elasticsearch bulk update"""

    # Status containers
    status_container = st.empty()
    progress_container = st.empty()
    stats_container = st.empty()

    status_container.info("Starting bulk update...")

    # Get initial count
    preview_count = elasticsearch_client.preview_update(namespace, only_missing)

    if preview_count == 0:
        status_container.success("🎉 No documents need updating!")
        return

    progress_bar = progress_container.progress(0)

    success_count = 0
    failed_count = 0

    def progress_callback(success, failed, total):
        nonlocal success_count, failed_count
        success_count = success
        failed_count = failed

        # Update progress
        if preview_count > 0:
            progress = min(total / preview_count, 1.0)
            progress_bar.progress(progress)

        stats_container.markdown(f"""
        **Progress:** {total:,} processed |
        **Success:** {success:,} |
        **Failed:** {failed}
        """)

    try:
        result = elasticsearch_client.bulk_add_clearance_level(
            namespace=namespace,
            only_missing=only_missing,
            progress_callback=progress_callback
        )

        progress_bar.progress(1.0)

        if result.get("error"):
            status_container.error(f"❌ Update completed with errors: {result['error']}")
        else:
            status_container.success(f"🎉 Update Complete! Updated {result['success']:,} documents")

        # Final stats
        st.markdown("---")
        st.subheader("📊 Update Results")

        col1, col2 = st.columns(2)
        col1.metric("Successfully Updated", f"{result['success']:,}")
        col2.metric("Failed", f"{result['failed']}")

        # Verify update
        st.markdown("---")
        st.subheader("✅ Verification")

        with st.spinner("Verifying update..."):
            if namespace:
                new_stats = elasticsearch_client.get_namespace_stats(namespace)
            else:
                new_stats = elasticsearch_client.get_all_stats()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", f"{new_stats.total_count:,}")
        col2.metric("With clearance_level", f"{new_stats.with_clearance_level:,}")
        col3.metric("Without clearance_level", f"{new_stats.without_clearance_level:,}")

    except Exception as e:
        status_container.error(f"❌ Update failed: {e}")
        st.exception(e)


def render_configuration():
    """Render the configuration page"""
    st.title("⚙️ Configuration")
    st.markdown("Current configuration settings for the migration tool.")

    # Pinecone Configuration
    st.subheader("🗄️ Pinecone Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Source Index (READ-ONLY)**")
        st.code(config.pinecone.old_index_name)
    with col2:
        st.markdown("**Target Index**")
        st.code(config.pinecone.new_index_name)

    st.markdown(f"**Environment:** `{config.pinecone.environment}`")

    # Gemini Configuration
    st.markdown("---")
    st.subheader("🤖 Gemini Embedding Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model**")
        st.code(config.gemini.model)
    with col2:
        st.markdown("**Output Dimensions**")
        st.code(str(config.gemini.output_dimensions))

    st.markdown(f"**Task Type:** `{config.gemini.task_type}`")

    # Elasticsearch Configuration
    st.markdown("---")
    st.subheader("🔎 Elasticsearch Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Endpoint**")
        st.code(config.elasticsearch.endpoint[:50] + "..." if len(config.elasticsearch.endpoint) > 50 else config.elasticsearch.endpoint)
    with col2:
        st.markdown("**Index**")
        st.code(config.elasticsearch.index_name)

    # Migration Configuration
    st.markdown("---")
    st.subheader("🔄 Migration Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query Batch Size", config.migration.query_batch_size)
    with col2:
        st.metric("Embedding Batch Size", config.migration.embedding_batch_size)
    with col3:
        st.metric("Upsert Batch Size", config.migration.upsert_batch_size)

    st.markdown(f"**Batch Delay:** `{config.migration.batch_delay}s`")

    # New Metadata
    st.markdown("---")
    st.subheader("🏷️ New Metadata (Added to All Vectors)")

    st.info(f"""
    **Key:** `{config.migration.new_metadata_key}`
    **Value:** `"{config.migration.new_metadata_value}"`

    This metadata will be added to every vector during migration.
    """)

    # Default Namespace
    st.markdown("---")
    st.subheader("📂 Default Namespace")
    st.code(config.migration.default_namespace)


def main():
    """Main application entry point"""
    init_session_state()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔍 Check Namespace":
        render_check_namespace()
    elif page == "🪞 Namespace Mirroring":
        render_mirroring()
    elif page == "🚀 Pinecone ReEmbedding":
        render_migration()
    elif page == "🔎 Elasticsearch Update":
        render_elasticsearch_update()
    elif page == "⚙️ Configuration":
        render_configuration()


if __name__ == "__main__":
    main()
