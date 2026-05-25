from semantic_kv.cli import simulate

if __name__ == "__main__":
    simulate(workload="basic", sessions=8, context=4096, decode_steps=64, policy="semantic")
