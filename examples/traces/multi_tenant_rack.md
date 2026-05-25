# multi_tenant_rack

Multiple tenants share a rack-scale inference system with overlapping and isolated prefixes.

## Expected semantic advantage

Rack-local prefix reuse, tenant-aware isolation, and avoided cross-rack movement.

## Assumptions

- synthetic multi-tenant rack
- cross_tenant_dedup_allowed=False
