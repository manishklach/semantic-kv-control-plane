# shared_enterprise

Many enterprise users share the same system prompt, policy block, tool schema, or RAG context.

## Expected semantic advantage

High prefix hit rate, high dedup savings, lower HBM pressure, and reduced repeated KV movement.

## Assumptions

- synthetic simulation
- exact-match prefix hash
