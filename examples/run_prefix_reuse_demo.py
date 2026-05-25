from semantic_kv.cli import compare

if __name__ == "__main__":
    compare(workload="shared-prefix", sessions=100, context=32768, decode_steps=128)
