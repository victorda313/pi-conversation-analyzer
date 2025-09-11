"""Azure Blob Storage helpers for loading instruction prompts.

Functions:
    get_blob_text(connection_string, container, blob_name) -> tuple[str, str | None]
"""
from __future__ import annotations
from typing import Optional, Tuple
from azure.storage.blob import BlobServiceClient


def get_blob_text(
    connection_string: str,
    container: str,
    blob_name: str,
) -> tuple[str, Optional[str]]:
    """Fetch a text blob and return its content and an identifier.

    The identifier is the blob's ETag, which we store as `instructions_version`
    for traceability in the database.

    Args:
        connection_string: Azure Storage connection string.
        container: Container name.
        blob_name: Blob name within the container.

    Returns:
        Tuple of (text_content, etag_or_none).
    """
    svc = BlobServiceClient.from_connection_string(connection_string)
    blob = svc.get_blob_client(container=container, blob=blob_name)
    downloader = blob.download_blob()
    text = downloader.readall().decode("utf-8")
    etag = getattr(downloader, "properties", None).etag if hasattr(downloader, "properties") else None
    return text, etag
