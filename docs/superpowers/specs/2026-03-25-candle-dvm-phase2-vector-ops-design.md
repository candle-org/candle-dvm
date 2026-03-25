# candle-dvm Phase 2 Vector Ops Design

Date: 2026-03-25
Status: Draft approved for implementation planning

## Summary

Phase 2 expands `candle-dvm` from a proof-of-concept host runtime that only supports `add` into a broader DVM-compatible vector execution layer. The project will continue to expose only operations that are fully implemented, encoded, and verified end-to-end on hardware. The initial Phase 2 strategy is to add ops in batches grouped by DVM ISA encoding structure, not by user-facing API names.

The goal is to grow the supported vector path while preserving the core guarantees established in phase 1:

- bytecode remains compatible with the existing DVM device-side VM
- hardware execution stays on the current 910B environment
- API exposure remains strict: only fully working ops are public
- each op is verified by both portable bytecode tests and 910B end-to-end tests

## Goals

### Primary goals

- Expand vector-path op coverage beyond `add`.
- Support both `fp32` and `fp16` variants where DVM has distinct ISA opcodes.
- Keep the public API aligned with actual implementation status.
- Maintain DVM bytecode compatibility and hardware execution correctness.
- Organize implementation by encoding families to reduce duplication and errors.

### Secondary goals

- Prepare the codebase for later support of broadcast, reduce, reshape, and additional vector utilities.
- Preserve clean layering between ISA encoding, op modeling, kernel codegen, and Python-facing API exposure.
- Keep Phase 2 additive: no redesign of the working phase-1 stack.

### Non-goals for this phase

- Cube kernel support.
- Mix kernel support.
- Candle dispatch integration.
- Dynamic shape support.
- 910A / 310B enablement.
- Zero-copy integration.

## Product rule

Phase 2 adopts a strict exposure rule:

> An op is only added to the public `Kernel` / `PyKernel` / decorator-facing API after its bytecode encoding, normalize behavior, dtype handling, and end-to-end hardware tests are all implemented and passing.

This means the public API intentionally lags internal scaffolding. Placeholder methods should not be exported for future ops.

## High-level architecture

The current phase-1 architecture remains intact:

- `isa.pyx` encodes DVM bytecode
- `code.pyx` owns the code buffer and relocations
- `system.pyx` loads runtime symbols, binaries, launches kernels, and manages memory
- `ops.pyx` models DVM graph objects and emits bytecode
- `kernel.pyx` performs xbuf assignment, codegen orchestration, and entry construction
- `api.pyx` exposes the graph builder
- `pykernel.py` exposes the end-to-end Python execution interface

Phase 2 extends only the vector-path portions of these files.

## Implementation strategy

### Chosen strategy: batch by encoding family

Ops will be added in batches grouped by DVM ISA encoding structure, because the encoding family determines:

- instruction word layout
- which fields go in the head word vs payload words
- whether workspace is required
- how dtype-specific opcode lookup works
- how much test structure can be reused

This is the most reliable way to scale bytecode-compatible implementation without duplicating fragile bit-packing logic.

## Batch plan

## Batch A: UnaryOp (`vUnary`, 2-word)

### Scope

Add unary ops that use the `vUnary` encoding family:

- `sqrt`
- `abs`
- `log`
- `exp`
- `round`
- `floor`
- `ceil`
- `trunc`
- `isfinite`
- `reciprocal`
- `logical_not`

### New internal type

- `UnaryOp(FlexOp)`

### Encoding shape

`vUnary` uses 2 words:

- `pc[0] = make_simd_head(opcode, xd, 2)`
- `pc[1] = xn << 32 | count`

Unlike `vBinary`, the head `ext` field is `xd`, not `xn`.

### Normalize behavior

- output shape = input shape
- output dtype = input dtype for most ops
- `isfinite` returns bool dtype

### Dtype routing

Ops route through `(unary_op, dtype) -> simd_opcode` tables.

For phase 2, both `fp32` and `fp16` are required when upstream DVM has both variants.

## Batch B: BinaryOp expansion (`vBinary`, 2-word)

### Scope

Extend the existing `BinaryOp` implementation beyond `add` to support:

- `sub`
- `mul`
- `div`
- `maximum`
- `minimum`

### Encoding shape

`vBinary` uses 2 words:

- `pc[0] = make_simd_head(opcode, xn, 2)`
- `pc[1] = count << 48 | xd << 18 | xm`

This matches the phase-1 `add` path.

### Dtype routing

All supported `fp32` / `fp16` variants use the same family but distinct opcodes.

## Batch C: BinaryScalarOp (`vBinaryS`, 2-word payload form)

### Scope

Add tensor-scalar arithmetic and extremum ops:

- `adds`
- `muls`
- `divs`
- `maxs`
- `mins`

This batch unlocks public API calls where one operand is a scalar.

### New internal type

- `BinaryScalarOp(FlexOp)`

### Encoding shape

`vBinaryS` uses 2 words:

- `pc[0] = make_simd_head(opcode, xn, 2)`
- `pc[1] = scalar_bits << 32 | compact_x(xd) << 16 | count`

Field widths are:

- `scalar_bits`: 32 bits, stored in bits `[32, 63]`
- `compact_x(xd)`: 13 bits, stored in bits `[16, 28]`
- `count`: 16 bits, stored in bits `[0, 15]`

Bit ranges in this spec are inclusive on both ends.

This matches upstream DVM `vBinaryS::Encode`. The scalar is encoded inline into the payload word.

### API rule

Public API methods like `add`, `mul`, `div`, `maximum`, `minimum` will route to `BinaryScalarOp` when one operand is scalar and the other is tensor.

## Batch D: CompareOp + CompareScalarOp

### Scope

Add comparison families:

- `equal`
- `not_equal`
- `greater`
- `greater_equal`
- `less`
- `less_equal`

for both tensor-tensor and tensor-scalar paths.

### New internal types

- `CompareOp(FlexOp)`
- `CompareScalarOp(FlexOp)`

### Encoding families

#### `vCompare`

Used for tensor-tensor compare. It includes workspace.

`vCompare` uses 2 words:

- `pc[0] = make_simd_head(opcode, type << 18 | xn, 2)`
- `pc[1] = count << 49 | compact_x(ws) << 36 | xd << 18 | xm`

Field layout (bit ranges inclusive on both ends):

- `type`: 4 bits, packed into bits `[40, 43]` of the 26-bit head `ext` field (ext bits `[18, 21]`)
- `xn`: 18 bits, packed into bits `[22, 39]` of the head `ext` field (ext bits `[0, 17]`)
- `count`: 15 bits, stored in bits `[49, 63]` of `pc[1]`
- `compact_x(ws)`: 13 bits, stored in bits `[36, 48]` of `pc[1]`
- `xd`: 18 bits, stored in bits `[18, 35]` of `pc[1]`
- `xm`: 18 bits, stored in bits `[0, 17]` of `pc[1]`

Note: `count` is 15 bits (not 16) in `vCompare`; this asymmetry with `vCompareS` (which uses 16 bits) matches the upstream DVM `isa.h` encoding comments exactly (`count(15) << 49` in vCompare vs `count(16) << 48` in vCompareS).

#### `vCompareS`

Used for tensor-scalar compare.

`vCompareS` uses 3 words:

- `pc[0] = make_simd_head(opcode, type << 18 | xn, 3)`
- `pc[1] = count << 48 | ws << 18 | xd`
- `pc[2] = scalar_bits`

Field layout (bit ranges inclusive on both ends):

- `type`: 4 bits in bits `[40, 43]` of the head `ext` field (ext bits `[18, 21]`)
- `xn`: 18 bits in bits `[22, 39]` of the head `ext` field (ext bits `[0, 17]`)
- `count`: 16 bits in bits `[48, 63]` of `pc[1]`
- `ws`: 18 bits in bits `[18, 35]` of `pc[1]`
- `xd`: 18 bits in bits `[0, 17]` of `pc[1]`

### Dtype behavior

Compare ops are dtype-routed. Different input dtypes use different opcodes:

- `(equal, fp32)` -> `V_CMP`
- `(equal, fp16)` -> `V_CMP_FP16`
- `(equal_scalar, fp32)` -> `V_CMPS`
- `(equal_scalar, fp16)` -> `V_CMPS_FP16`

The same pattern applies to all six comparison types (equal, not_equal, greater, greater_equal, less, less_equal). The compare `type` field (`V_CMP_EQ`, `V_CMP_NE`, etc.) is encoded into the instruction ext field, while the dtype-specific opcode selects the right SIMD function in the device VM.

- input dtype follows tensor input
- output dtype is bool

### Kernel impact

This is the first batch that requires the kernel layer to reserve extra temporary workspace/xbuf slots beyond direct result buffers.

## Batch E: SelectOp (`vSelect`)

### Scope

Add:

- `select(cond, lhs, rhs)`

### New internal type

- `SelectOp(FlexOp)`

### Encoding

`vSelect` uses 3 words and workspace.

### Semantics

- `cond` must be a bool tensor
- `lhs` and `rhs` must be shape-compatible and dtype-compatible
- output shape follows the data operands

### Kernel impact

Requires workspace-capable xbuf assignment.

## Batch F: CopyOp + CastOp

### Scope

Add:

- `copy`
- `cast`

### New internal types

- `CopyOp(FlexOp)`
- `CastOp(FlexOp)`

`CopyOp` and `CastOp` inherit from `FlexOp` for consistency with the rest of the phase-2 vector op family. Even though they do not require workspace in the first implementation, this keeps normalize/emit plumbing uniform and avoids introducing a separate inheritance rule for similar op-shaped nodes.

### Encoding

- `CopyOp` uses `vCopy`
- `CastOp` uses unary-style specialized cast opcodes

### Dtype scope

For phase 2, cast support should focus on the combinations already most useful in the current vector path:

- `fp32 -> fp16`
- `fp16 -> fp32`
- `fp32 -> int32`
- `int32 -> fp32`

Additional cast combinations can be added only when upstream DVM opcodes and tests are available.

## Batch G: BroadcastOp

### Scope

Add vector-path broadcast operations:

- scalar broadcast / `full`
- tensor broadcast along supported axes

### New internal type

- `BroadcastOp(FlexOp)`

### Encoding families

Broadcast uses distinct families depending on broadcast mode:

- scalar broadcast
- x-direction broadcast
- y-direction broadcast

### Complexity note

This batch is where shape semantics begin to matter more than in phase 1. Even though the bytecode is still vector-path, shape validation and output-shape inference must be more careful than in unary/binary batches.

## Batch H: ReduceOp

### Scope

Add:

- `sum`
- `max`
- `min`

for the supported vector reduction forms.

### New internal type

- `ReduceOp(FlexOp)`

### Complexity note

This is the most complex Phase 2 batch because upstream DVM uses distinct encoding families for different reduction shapes/directions. It should come after unary, binary, scalar, compare, select, copy/cast, and broadcast are already stable.

## Batch I: Remaining vector utilities

### Scope

Lower-priority ops that are still part of the vector path:

- `reshape`
- `one_hot`
- `element_any`
- any remaining small utility ops justified by upstream support

These can be scheduled only after the earlier batches are stable.

## Common infrastructure changes

## `isa.pyx`

Phase 2 extends `isa.pyx` with additional encode helpers and opcode routing support.

### Required additions

- unary opcode tables
- binary scalar opcode tables
- compare opcode tables
- compare-scalar opcode tables
- copy opcode tables
- cast opcode tables
- encode helpers for each supported instruction struct family

### Principle

Each encoding helper should map directly to a single upstream DVM instruction struct layout. Do not mix multiple families into one generic helper.

## `ops.pyx`

Phase 2 extends `ops.pyx` with new cdef classes for each op family.

### Principle

Keep one cdef class per semantic op family, not one giant switch statement. This makes normalize and emit behavior easier to reason about and test.

## `kernel.pyx`

Phase 2 requires one significant kernel-level expansion:

### Workspace-capable xbuf assignment

Phase 1 only needed direct result slots. Phase 2 needs workspace-aware assignment because compare/select and later reduce families use temporary buffers.

The simplest acceptable approach is still monotonic allocation, but it must distinguish:

- result xbufs
- workspace xbufs

### Required protocol between ops and kernel

Each workspace-using op must expose its workspace requirement explicitly to the kernel layer.

The required protocol is:

- `FlexOp` gains a `workspace_slots()` method returning an integer count of extra xbuf slots required
- default implementation returns `0`
- `CompareOp`, `CompareScalarOp`, and `SelectOp` override it to return `1`
- later reduce ops may return more than `1` depending on encoding family needs

The kernel codegen path queries `workspace_slots()` during xbuf assignment and allocates those slots after assigning the op's result slot.

There is no need to implement liveness-based reuse yet.

## `api.pyx`

Phase 2 extends the public `Kernel` API only for ops that are fully complete.

### Rule

Each batch lands together:

1. internal op class exists
2. bytecode emit works
3. portable tests pass
4. hardware tests pass
5. public API method is exported

No earlier.

### Error propagation contract

The public API must not hide kernel-layer resource errors.

When a workspace-using op is invoked:

- normalize-time shape or dtype mismatches raise `ValueError`
- emit-time unsupported opcode/dtype pairs raise `NotImplementedError`
- kernel-layer inability to allocate required workspace slots raises a runtime error that propagates unchanged through `api.pyx`

The `Kernel` API should remain a thin graph-building surface, not an error translation layer.

## `pykernel.py`

The decorator-facing API should mirror `api.pyx` additions only after the same op batch is complete.

## Dtype strategy

Phase 2 supports `fp32` and `fp16` together.

### Routing model

Every op family should use explicit `(semantic_op, dtype) -> opcode` tables.

If an `(op, dtype)` pair is not available, emit should raise `NotImplementedError` rather than guessing or silently falling back.

## Testing strategy

Every batch must include two layers of verification.

### Portable tests

These validate:

- normalize behavior
- dtype propagation
- shape propagation
- emitted instruction word count
- exact opcode/field placement where stable
- relocation count when relevant
- opcode routing table coverage: every `(op, dtype)` pair added in a batch must have a portable test asserting it maps to the expected opcode constant before any hardware execution is attempted

### Hardware tests

These validate:

- actual 910B execution
- `fp32` correctness vs NumPy
- `fp16` correctness vs NumPy with relaxed tolerance
- no regressions in the end-to-end decorator flow

## Public exposure policy tests

Add explicit tests that verify unsupported ops are not publicly exported yet. This makes the “only expose what works” rule enforceable, not just aspirational.

## Suggested batch order

The recommended delivery order is:

1. Batch A: unary
2. Batch B: binary expansion
3. Batch C: binary scalar
4. Batch D: compare + compare scalar
5. Batch E: select
6. Batch F: copy + cast
7. Batch G: broadcast
8. Batch H: reduce
9. Batch I: remaining vector utilities

This order goes from the simplest shared encoding families to the most shape- and workspace-sensitive ones.

## Error handling

Phase 2 should continue the phase-1 rule of explicit failure over silent fallback.

Examples:

- unsupported dtype/opcode pair -> `NotImplementedError`
- unsupported shape semantics in a partially completed batch -> `NotImplementedError`
- compare/select/reduce workspace allocation mismatch -> explicit runtime error

## Recommendation

Proceed with batch-by-encoding implementation, expose ops only after full completion, and require both `fp32` and `fp16` validation before public API exposure. This gives `candle-dvm` a disciplined and scalable growth path while keeping its public surface trustworthy.
