"""
PDF ingestion script: load, chunk, embed, and upload documents to Pinecone.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def validate_env() -> tuple[str, str]:
    """Validate required environment variables and return (api_key, index_name)."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    missing = []
    if not api_key:
        missing.append("PINECONE_API_KEY")
    if not index_name:
        missing.append("PINECONE_INDEX_NAME")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your credentials."
        )

    return api_key, index_name


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_pdf_documents(pdf_paths: list[Path]) -> list:
    """Load PDF files using PyPDFLoader."""
    documents = []
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
        documents.extend(docs)
        print(f"  Loaded: {pdf_path.name} ({len(docs)} pages)")

    return documents


def split_documents(documents: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def ingest_pdfs(pdf_paths: list[Path], index_name: str | None = None) -> int:
    """
    Ingest PDF files into an existing Pinecone index.

    Returns the number of chunks uploaded.
    """
    api_key, default_index = validate_env()
    target_index = index_name or default_index

    if not pdf_paths:
        raise ValueError("No PDF files provided for ingestion.")

    print(f"Ingesting {len(pdf_paths)} PDF(s) into index '{target_index}'...")

    documents = load_pdf_documents(pdf_paths)
    if not documents:
        raise ValueError("No content extracted from the provided PDF files.")

    chunks = split_documents(documents)
    print(f"  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embeddings = get_embeddings()

    Pinecone(api_key=api_key)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=target_index,
    )

    print(f"Successfully uploaded {len(chunks)} vectors to Pinecone.")
    return len(chunks)


def collect_pdfs_from_directory(directory: Path) -> list[Path]:
    """Collect all PDF files from a directory."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF policy documents into Pinecone for RAG retrieval."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dir",
        type=str,
        help="Directory containing PDF files to ingest",
    )
    group.add_argument(
        "--file",
        type=str,
        nargs="+",
        help="One or more PDF file paths to ingest",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Pinecone index name (overrides PINECONE_INDEX_NAME env var)",
    )

    args = parser.parse_args()

    try:
        if args.dir:
            pdf_paths = collect_pdfs_from_directory(Path(args.dir))
        else:
            pdf_paths = [Path(f) for f in args.file]

        ingest_pdfs(pdf_paths, index_name=args.index)
    except (EnvironmentError, FileNotFoundError, ValueError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error during ingestion: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
