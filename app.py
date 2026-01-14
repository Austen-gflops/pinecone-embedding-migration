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
            ["📊 Dashboard", "🔍 Check Namespace", "🚀 Run Migration", "🔎 Elasticsearch Update", "⚙️ Configuration"],
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

    # Connection test
    st.subheader("🔗 Connection Status")
    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Testing connection..."):
            connected = elasticsearch_client.test_connection()

        if connected:
            st.success("✅ Connected to Elasticsearch")
        else:
            st.error("❌ Connection failed")
            return

    with col2:
        stats = elasticsearch_client.get_index_stats()
        st.metric("Index", config.elasticsearch.index_name)
        st.metric("Total Documents", f"{stats.get('doc_count', 0):,}")

    # Overall stats
    st.markdown("---")
    st.subheader("📊 Overall Statistics")

    with st.spinner("Fetching statistics..."):
        all_stats = elasticsearch_client.get_all_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", f"{all_stats.total_count:,}")
    col2.metric("With clearance_level", f"{all_stats.with_clearance_level:,}")
    col3.metric("Without clearance_level", f"{all_stats.without_clearance_level:,}")

    if all_stats.without_clearance_level == 0 and all_stats.total_count > 0:
        st.success("🎉 All documents already have clearance_level!")

    # Namespace selection
    st.markdown("---")
    st.subheader("🏷️ Update by Namespace")

    # Get available namespaces
    with st.spinner("Fetching namespaces..."):
        namespaces = elasticsearch_client.list_all_namespaces()

    if not namespaces:
        st.warning("No namespaces found in the index.")
        return

    namespace_options = ["All Namespaces"] + namespaces
    selected_namespace = st.selectbox(
        "Select Namespace",
        namespace_options,
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
    elif page == "🚀 Run Migration":
        render_migration()
    elif page == "🔎 Elasticsearch Update":
        render_elasticsearch_update()
    elif page == "⚙️ Configuration":
        render_configuration()


if __name__ == "__main__":
    main()
