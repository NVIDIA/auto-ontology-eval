#!/bin/bash
# Activate the NeMo Retriever virtual environment
# Run: source unstructured/benchmark/setup.sh
#
# NeMo Retriever pins versions that clash with this repo's own environment, so
# it keeps the separate venv it was installed into. Override RETRIEVER_VENV if
# yours lives elsewhere.

RETRIEVER_VENV="${RETRIEVER_VENV:-$HOME/nemo-retriever-rag/retriever}"

if [ ! -f "$RETRIEVER_VENV/bin/activate" ]; then
    echo "No NeMo Retriever venv at $RETRIEVER_VENV."
    echo "Set RETRIEVER_VENV to its location and re-source this file."
    return 1 2>/dev/null || exit 1
fi

source "$RETRIEVER_VENV/bin/activate"
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "NeMo Retriever env activated. Python: $(python --version)"
echo ""
echo "Set your API key:  export NVIDIA_API_KEY=your_key_here  (or put it in the repo-root .env)"
echo "Run the benchmark: python $BENCH_DIR/run.py --limit 10"
echo "Re-score only:     python $BENCH_DIR/run.py --report-only"
echo "Re-ingest the PDFs:python $BENCH_DIR/ingest.py"
echo "Ask a question:    python $BENCH_DIR/main.py 'your question'"
echo "List stored docs:  python $BENCH_DIR/remove.py --list"
