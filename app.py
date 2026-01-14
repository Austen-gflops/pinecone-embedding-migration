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


def init_session_state():
    """Initialize session state variables"""
    if 'migration_running' not in st.session_state:
        st.session_state.migration_running = False
    if 'migration_stats' not in st.session_state:
        st.session_state.migration_stats = None
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None


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
            ["📊 Dashboard", "🔍 Check Namespace", "⚖️ Compare Indexes", "🚀 Run Migration", "🔧 Fix Metadata", "⚙️ Configuration"],
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


def render_compare_indexes():
    """Render the index comparison page"""
    st.title("⚖️ Compare Indexes")
    st.markdown("Compare namespace contents between source and target indexes.")

    namespace = st.text_input(
        "Namespace ID",
        value=config.migration.default_namespace,
        help="Enter the namespace UUID to compare"
    )

    if st.button("⚖️ Compare", type="primary"):
        if not namespace:
            st.error("Please enter a namespace ID")
            return

        with st.spinner("Fetching comparison data..."):
            source_stats = pinecone_client.get_namespace_stats(
                config.pinecone.old_index_name, namespace
            )
            target_stats = pinecone_client.get_namespace_stats(
                config.pinecone.new_index_name, namespace
            )

        # Comparison table
        st.subheader("📊 Comparison Results")

        comparison_data = {
            "Property": ["Index Name", "Vector Count", "Namespace Exists"],
            "Source": [
                config.pinecone.old_index_name,
                f"{source_stats.get('vector_count', 0):,}",
                "✅ Yes" if source_stats.get("exists") else "❌ No"
            ],
            "Target": [
                config.pinecone.new_index_name,
                f"{target_stats.get('vector_count', 0):,}",
                "✅ Yes" if target_stats.get("exists") else "❌ No"
            ]
        }

        st.table(comparison_data)

        # Migration status
        source_count = source_stats.get('vector_count', 0)
        target_count = target_stats.get('vector_count', 0)
        remaining = source_count - target_count

        st.markdown("---")
        st.subheader("📈 Migration Status")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total in Source", f"{source_count:,}")
        col2.metric("Migrated to Target", f"{target_count:,}")
        col3.metric("Remaining", f"{remaining:,}")

        if remaining > 0:
            st.warning(f"⚠️ {remaining:,} vectors need to be migrated")
        elif source_count > 0:
            st.success("🎉 All vectors have been migrated!")


def render_migration():
    """Render the migration page"""
    st.title("🚀 Run Migration")
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


def render_fix_metadata():
    """Render the Fix Metadata page for updating clearance_level from string to number"""
    st.title("🔧 Fix Metadata")
    st.markdown("""
    **Purpose:** Update the `clearance_level` metadata field from string `"1"` to number `1`.

    This operation:
    - ✅ Only updates metadata (no re-embedding)
    - ✅ Preserves all other metadata fields
    - ✅ Uses Pinecone's bulk update API for maximum efficiency
    - ✅ Single API call updates all matching vectors
    """)

    st.markdown("---")

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        namespace = st.text_input(
            "Namespace",
            value=config.migration.default_namespace,
            help="The namespace containing vectors to fix"
        )

        st.markdown(f"**Target Index:** `{config.pinecone.new_index_name}`")

    with col2:
        st.info("""
        **Bulk Update API:**
        Uses Pinecone's optimized bulk update API that:
        - Filters vectors by metadata
        - Updates all matching vectors in a single API call
        - Only updates the **target** index (askdona)
        """)

    st.markdown("---")

    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        scan_button = st.button("🔍 Scan (Preview)", type="secondary", use_container_width=True)

    with col_btn2:
        dry_run_button = st.button("🧪 Dry Run", type="secondary", use_container_width=True)

    with col_btn3:
        bulk_fix_button = st.button("🚀 Bulk Fix (Optimized)", type="primary", use_container_width=True)

    # Always update target index only (source is read-only)
    is_source = False
    index_name = config.pinecone.new_index_name

    # Scan operation (preview only)
    if scan_button:
        if not namespace:
            st.error("Please enter a namespace")
            return

        with st.spinner(f"Scanning {index_name} namespace: {namespace}..."):
            try:
                # Get vectors with string clearance_level
                vectors_to_fix = pinecone_client.get_vectors_with_string_clearance(
                    namespace=namespace,
                    source=is_source,
                    max_workers=5
                )

                if vectors_to_fix:
                    st.success(f"Found **{len(vectors_to_fix)}** vectors with string clearance_level that need fixing.")

                    # Show sample IDs
                    with st.expander("Sample Vector IDs (first 10)"):
                        for vec_id in vectors_to_fix[:10]:
                            st.code(vec_id)
                else:
                    st.info("No vectors found with string clearance_level. All vectors are already using number format.")

            except Exception as e:
                st.error(f"Error scanning: {e}")

    # Dry run operation
    if dry_run_button:
        if not namespace:
            st.error("Please enter a namespace")
            return

        st.markdown("---")
        st.subheader("🧪 Dry Run Results")

        with st.spinner(f"Running dry run on {index_name}..."):
            try:
                result = pinecone_client.bulk_update_metadata_by_filter(
                    namespace=namespace,
                    metadata_filter={"clearance_level": {"$eq": "1"}},
                    metadata_updates={"clearance_level": 1},
                    source=is_source,
                    dry_run=True
                )

                if result["success"]:
                    st.success(f"Dry run complete! **{result['updated_count']}** vectors would be updated.")
                    with st.expander("API Response"):
                        st.json(result["response"])
                else:
                    st.error(f"Dry run failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Error in dry run: {e}")

    # Bulk fix operation (optimized)
    if bulk_fix_button:
        if not namespace:
            st.error("Please enter a namespace")
            return

        st.markdown("---")
        st.subheader("🚀 Bulk Update Progress")

        status_text = st.empty()
        log_container = st.container()

        with log_container:
            st.markdown("### Update Log")
            log_expander = st.expander("Detailed Log", expanded=True)

        try:
            with log_expander:
                st.write(f"Starting bulk metadata update...")
                st.write(f"Index: {index_name}")
                st.write(f"Namespace: {namespace}")
                st.write(f"Filter: clearance_level == \"1\" (string)")
                st.write(f"Update: clearance_level = 1 (integer)")
                st.write("---")
                st.write("Using Pinecone Bulk Update API (single API call)...")

            status_text.text("Executing bulk update...")

            # Perform the bulk update using filter
            result = pinecone_client.bulk_update_metadata_by_filter(
                namespace=namespace,
                metadata_filter={"clearance_level": {"$eq": "1"}},
                metadata_updates={"clearance_level": 1},
                source=is_source,
                dry_run=False
            )

            status_text.text("Complete!")

            with log_expander:
                st.write("---")
                if result["success"]:
                    st.write(f"✅ Successfully updated: {result['updated_count']} vectors")
                else:
                    st.write(f"❌ Update failed: {result.get('error', 'Unknown error')}")

            # Show results
            st.markdown("---")
            st.subheader("✅ Bulk Fix Complete" if result["success"] else "❌ Bulk Fix Failed")

            if result["success"]:
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Vectors Updated", result["updated_count"])
                with col_res2:
                    st.metric("API Calls", "1 (bulk)")

                if result["updated_count"] > 0:
                    st.success(f"🎉 Successfully updated {result['updated_count']} vectors in a single API call!")
                    st.balloons()
                else:
                    st.info("No vectors matched the filter. All vectors may already have integer clearance_level.")

                with st.expander("API Response Details"):
                    st.json(result.get("response", {}))
            else:
                st.error(f"Update failed: {result.get('error', 'Unknown error')}")
                if result.get("status_code"):
                    st.write(f"HTTP Status Code: {result['status_code']}")

        except Exception as e:
            st.error(f"Error during bulk fix: {e}")


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
    elif page == "⚖️ Compare Indexes":
        render_compare_indexes()
    elif page == "🚀 Run Migration":
        render_migration()
    elif page == "🔧 Fix Metadata":
        render_fix_metadata()
    elif page == "⚙️ Configuration":
        render_configuration()


if __name__ == "__main__":
    main()
