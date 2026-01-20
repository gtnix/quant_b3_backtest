# Quality Bar

Global quality contract for all Quant Finance Team skills.

---

## SKILL.md Structure (Required)

Every skill must follow this structure:

```yaml
---
name: <skill-id>
description: "<one-line purpose>"
triggers:
  - command: "/<skill-id>"
domain_knowledge:
  - <key expertise area 1>
  - <key expertise area 2>
---
```

**Mandatory sections after frontmatter:**

1. **Role** - One-line persona definition
2. **Expertise Map** - Technical domains with depth indicators
3. **When to Use / When NOT to Use** - Clear scope boundaries
4. **Operating Rules** - Hard constraints (numbered, imperative)
5. **Repo Anchors** - Exact file paths this skill must consult
6. **Deliverables** - Checklists, templates, artifacts
7. **Acceptance Criteria** - Objective pass/fail conditions
8. **Failure Modes** - Common traps to avoid
9. **Collaboration Hooks** - Handoff to other skills

---

## Language Standards

- **Tone**: Direct, technical, zero fluff
- **Voice**: Imperative for rules, declarative for context
- **Code**: Max 80 chars/line, always include file paths
- **Numbers**: Use exact values, never "about" or "around"
- **Promises**: Never promise specific speedups without measured baseline

---

## Permissions

**ALLOWED:**
- Suggest changes with exact repo paths
- Create benchmarks and profiling scripts
- Propose architectural refactors
- Reference external documentation with URLs

**FORBIDDEN:**
- Invent files or functions that don't exist
- Promise performance gains without baseline measurement
- Modify numerical behavior (must be bit-exact)
- Skip validation steps to save time

---

## Definition of Done

### PR Checklist (Performance)
- [ ] Benchmark before: `cargo bench --save-baseline before`
- [ ] Benchmark after: `cargo bench --baseline before`
- [ ] Regression check: <= 5% vs baseline (per PERFORMANCE_CONTRACT.md)
- [ ] Determinism: 3 consecutive runs produce identical output
- [ ] No new allocations in hot path (validate with dhat or perf)

### Reproducibility Requirements
- Git commit hash recorded
- Seed value documented (if randomized)
- Environment specs: CPU, RAM, Rust version
- Config snapshot: exact TOML used

### Artifact Naming
```
{run_id}_{date}_{scenario}.{ext}
```
Example: `a1b2c3d4_20260118_10assets_252d.json`

---

## Anti-Hallucination Rules

1. **Always verify paths exist** before referencing them
2. **Quote exact function signatures** when suggesting changes
3. **Never invent APIs** - check crate docs first
4. **Cite line numbers** when discussing specific code
5. **If unsure, say so** - "I need to verify X before proceeding"

### Before Suggesting Any Change:
```
1. Read the target file
2. Confirm the function/struct exists
3. Understand current behavior
4. Only then propose modification
```

---

## Quality Gates by Skill Type

| Skill Type | Primary Gate | Secondary Gate |
|------------|--------------|----------------|
| Performance | Benchmark delta | Zero regression |
| Strategy | Sharpe consistency | OOS validation |
| Risk | Constraint satisfaction | Stress test pass |
| Data | Schema validation | Integrity check |
