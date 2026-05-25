from semantic_kv.cli import compare

if __name__ == "__main__":
    compare(workload="agentic-tool", sessions=32, context=8192, decode_steps=64)
