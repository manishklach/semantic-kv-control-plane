# long_context

Few users with very large contexts and cold historical ranges.

## Expected semantic advantage

Compression and tiering reduce HBM pressure; generic spill can increase stall proxy.

## Assumptions

- synthetic long-context trace
- cold_prefix_ratio=0.7
