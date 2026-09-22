That is a better target: borrow PyTorch’s proven separation of metadata, storage, and allocation, but make different choices where PyTorch carries legacy or ecosystem complexity.

“70% production-ready” is not measurable as a percentage, but we can target the important properties:

- Memory-safe ownership
- Shared storage and views
- Multiple devices
- Pluggable allocation
- Stream-correct CUDA execution
- Runtime dtype dispatch
- Useful diagnostics
- Serialization
- Testability
- Extensibility without PyTorch’s backend breadth

## Bumblebee Architecture

Use four primary layers:

```text
Tensor
    value-owned metadata
    shared Storage

Storage
    owns Allocation
    tracks capacity and mutation version

Allocation
    opaque pointer
    memory location
    allocation provenance

MemoryResource
    creates and destroys allocations
```

Operations consume checked typed views and an explicit execution context:

```text
Tensor
    -> checked TensorView<T, MemorySpace>
    -> operation
    -> ExecutionContext
    -> CPU implementation or CUDA kernel
```

## What Makes It Ours

### Value-Owned Metadata

PyTorch’s `Tensor` is a handle to `TensorImpl`. Bumblebee can store metadata directly in `Tensor`:

```text
Tensor:
    TensorLayout
    shared Storage
    optional gradient reference
```

Copying a tensor copies its small metadata and shares only storage.

This makes view semantics straightforward:

```text
original:
    layout A
    storage S

transpose:
    layout B
    storage S
```

There is no shared mutable `TensorImpl` layer. Metadata changes produce a new layout or explicitly mutate that particular tensor object.

Benefits:

- One less allocation and indirection
- Easier debugging
- Natural value semantics for shape and strides
- Views share only what must be shared
- Fewer hidden aliasing relationships

### Explicit Execution Context

Avoid relying primarily on global “current CUDA device” and “current stream” state.

Operations receive an execution context containing:

```text
device
CUDA stream
allocator/resource
synchronization policy
diagnostic hooks
```

Conceptually:

```text
add(context, destination, left, right)
copy(context, destination, source)
```

Convenience overloads can use a default context, but the implementation remains explicit.

This is a strong production feature because it makes:

- Multiple streams understandable
- Multi-GPU behavior explicit
- Tests deterministic
- Allocator selection local
- Asynchronous execution traceable
- Future graph capture easier

### Checked Typed Views

Tensor storage remains runtime-typed, but kernels receive typed, memory-space-aware views:

```text
TensorView<float, CPU>
TensorView<float, CUDA>
TensorView<const float, CUDA>
```

Creating a view validates:

- Tensor dtype matches the requested type
- Tensor device matches the memory space
- Storage bounds cover the layout
- Alignment is sufficient
- Writable access is permitted

The view contains:

```text
typed pointer
shape
strides
storage offset
```

This is safer than passing raw pointers throughout the operation layer and lighter than PyTorch’s general dispatcher machinery.

## Core Types

### Device

Represent:

```text
DeviceType:
    CPU
    CUDA

Device:
    type
    index
```

Properties:

- `CPU` normally has index zero or no index.
- CUDA indices identify physical devices.
- Allocation records the exact device.
- Operations reject mismatched devices unless transfer is explicit.

### DType

Use a stable runtime descriptor:

```text
identifier
name
size
alignment
category
```

Initially support:

- `Float32`
- `Float16`
- `Int32`
- `UInt8`
- `Bool`

Add types according to actual kernel requirements.

Avoid encoding arbitrary C++ type information in serialized tensors. Serialized dtype identifiers need stable framework-defined meanings.

### TensorLayout

`TensorLayout` owns:

```text
shape
strides
storage offset
dtype
cached element count
cached layout flags
```

It should be independently validatable.

Layout flags may include:

```text
contiguous
non-overlapping and dense
broadcasted
has zero elements
```

Store shape and strides in a small-vector-like container so common ranks avoid heap allocation.

Metadata rules:

- Strides are measured in elements.
- Storage offset is measured in elements.
- Rank equals shape length and stride length.
- Arithmetic is checked for overflow.
- Reachable storage bounds are validated.
- Rank-zero shape represents a scalar.
- Any zero-sized dimension produces zero elements.

### Allocation

`Allocation` is move-only and owns one allocation:

```text
pointer
allocated bytes
device
owning resource
allocation identifier
```

It should not know about shape, dtype, or tensors.

The allocation identifier is useful for:

- Diagnostics
- Profiling
- Detecting alias relationships
- Logging transfers
- Memory snapshots

Cleanup must be non-throwing.

### MemoryResource

Use a byte-oriented resource interface:

```text
allocate(request) -> Allocation
deallocate(allocation)
```

An allocation request contains:

```text
byte count
alignment
device
stream
usage category
```

Initial resources:

```text
HostMemoryResource
CudaMemoryResource
```

Later resources:

```text
PinnedHostMemoryResource
CudaPoolMemoryResource
CountingMemoryResource
DebugMemoryResource
```

A resource should have stable lifetime, normally through shared ownership or application-level registration. Storage must never retain a dangling resource reference.

### Storage

Storage owns the allocation and storage-level state:

```text
Allocation
logical byte size
capacity
resizable flag
mutation version
```

Multiple tensors can share Storage.

Storage does not know:

- Shape
- Strides
- Dtype
- Autograd graph
- Operation semantics

The mutation version is shared by aliases. An in-place write increments it, allowing autograd and debugging tools to detect stale assumptions.

## Tensor Semantics

A `Tensor` contains:

```text
TensorLayout
shared Storage
optional GradState reference
```

Copy behavior:

```text
copy Tensor handle:
    copy metadata
    share Storage

clone:
    allocate new Storage
    copy elements
```

View behavior:

```text
reshape when compatible:
    new TensorLayout
    same Storage

transpose:
    permuted shape and strides
    same Storage

slice:
    adjusted shape and offset
    same Storage
```

In-place metadata operations modify only that tensor’s value-owned layout. In-place data operations affect every alias because storage is shared.

## Storage Bounds

Every tensor must prove that its layout fits inside storage.

For a contiguous tensor:

```text
required bytes = numel * element size
```

For a strided tensor, calculate the reachable range from:

- Storage offset
- Dimension extents
- Strides
- Element size

Initially reject negative strides. They complicate lowest-address calculation and are unnecessary for the transformer.

Broadcast strides of zero can be introduced later, but writable broadcast views should generally be rejected because multiple logical elements alias the same address.

## Operation Model

Use a lightweight dispatcher rather than PyTorch dispatch keys.

An operation proceeds through:

```text
schema validation
shape inference
dtype resolution
device selection
output allocation
typed view creation
backend implementation
```

Dispatch key can initially be:

```text
operation
device type
dtype
layout class
```

For the first implementation, explicit switches are adequate. Once several operations repeat the same dispatch logic, extract a registry or generated table.

Do not build a global plugin system until there is a real external backend.

## ExecutionContext

Execution context should include:

```text
Device
CUDA stream when applicable
MemoryResource
synchronization mode
diagnostic sink
```

CPU contexts may later include:

```text
thread pool
vectorization capabilities
scratch allocator
```

Every asynchronous CUDA operation is associated with a context. This prevents stream behavior from becoming implicit and scattered.

## Error Model

Replace printing and raw error strings with structured errors.

Error categories:

```text
invalid shape
shape overflow
dtype mismatch
device mismatch
unsupported operation
allocation failure
storage bounds violation
CUDA runtime failure
serialization failure
```

Choose either:

- Exceptions at public boundaries
- Result/status objects throughout

Do not inconsistently mix printing, exceptions, and nullable results.

For a C++ tensor library, exceptions for construction and validation combined with non-throwing destructors are reasonable.

## Autograd

Keep autograd outside the primary storage architecture.

Conceptually:

```text
Tensor:
    optional GradState reference

GradState:
    requires_grad
    accumulated gradient
    producing node
    expected storage version

AutogradNode:
    input edges
    backward operation
    saved tensors
```

This differs from tightly embedding broad autograd behavior into the tensor implementation.

Saved tensors record storage versions. If an aliased tensor is modified in place, backward can detect that the saved value is invalid.

Do not implement autograd until CPU and CUDA forward operations are stable.

## Serialization

Define a Bumblebee tensor format around logical values, not allocation internals:

```text
format version
dtype
shape
strides or contiguous marker
storage offset
payload byte order
payload
```

Normally serialize payload into a device-independent CPU representation. A CUDA tensor is copied to host during saving and restored to a requested device during loading.

Never serialize:

- Raw addresses
- Resource pointers
- CUDA stream handles
- Deleter functions

## Diagnostics

Production readiness benefits from observability early.

Track:

```text
allocation ID
allocated bytes
device
resource name
creation site in debug builds
live alias count
storage mutation version
```

Allow diagnostics to be disabled in optimized builds.

A wrapping memory resource can report:

- Current live bytes
- Peak bytes
- Allocation count
- Cache hit rate
- Out-of-memory context

## Directory Structure

A coherent layout would be:

```text
include/bumblebee/core/
    device.h
    dtype.h
    error.h
    execution_context.h

include/bumblebee/memory/
    allocation.h
    memory_resource.h
    host_memory_resource.h
    cuda_memory_resource.h
    storage.h

include/bumblebee/tensor/
    tensor_layout.h
    tensor_view.h
    tensor.h
    tensor_options.h

include/bumblebee/ops/
    copy.h
    elementwise.h
    reduction.h
```

Implementation files can mirror that structure under `src/bumblebee`.

## Migration From Current Buffers

Do not discard the buffer work. It established the required RAII behavior.

Progression:

1. Extract their allocation logic into host and CUDA memory resources.
2. Introduce `Allocation` as the move-only returned owner.
3. Introduce shared `Storage`.
4. Let `Tensor` contain layout plus shared storage.
5. Retire the buffers or keep them as thin convenience wrappers around `Allocation`.
6. Ensure there is only one ownership authority for every pointer.

Avoid having both a buffer and `Allocation` believe they own the same pointer.

## Practical Scope

This architecture can credibly support:

- CPU and multi-GPU CUDA tensors
- Shared storage and strided views
- Runtime dtypes
- Pluggable and pooled allocation
- Explicit asynchronous streams
- Typed CPU and CUDA kernels
- Autograd version safety
- Serialization
- Instrumentation

It intentionally avoids PyTorch’s largest complexity sources:

- Dozens of device backends
- Symbolic shape systems
- Sparse and quantized tensor subclasses
- Python compatibility history
- Generated operator schemas
- Distributed tensor subclasses
- Multiple compiler stacks
- Mobile compatibility layers
- Global dispatch-key composition

The defining Bumblebee design would be:

```text
value-owned tensor metadata
shared storage allocation
resource-based memory management
checked typed memory-space views
explicit execution contexts
lightweight operation dispatch
```

That is recognizably informed by PyTorch, but it is not merely PyTorch with names changed.
