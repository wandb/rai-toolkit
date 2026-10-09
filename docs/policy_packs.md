# Policy packs

A policy pack is a versioned directory of YAML policies owned by one team. Packs let organizations encode their own governance, compliance, risk, and approval requirements and run them alongside the toolkit's built-in framework mappings (NIST AI RMF, EU AI Act, MIT AI Risk Repository).

## Layout

```
my-policy-pack/
  pack.yaml          # manifest: name, version, owner, frameworks
  data-residency.yaml
  approval-language.yaml
```

## Run a pack alongside the built-ins

```bash
rai assess my_pkg.MyModel --preset financial --policies-dir my-policy-pack/
```

## Lint it in CI

Policy changes should be reviewed like code:

```bash
rai policies lint my-policy-pack/
```

## Working example

See `rai_toolkit/policies/packs/example_enterprise_pack/` for a working example: a `pack.yaml` manifest plus two organization-specific policies that run next to the 13 starter policies.

This convention was requested in issue #3.
