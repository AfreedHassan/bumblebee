`Allocation` should be a general, mechanism-neutral ownership type. `ArenaAllocator` should be one possible allocation strategy that produces `Allocation` objects.

They solve different problems:

```text
Allocation:
    owns one memory region

MemoryResource:
    decides how regions are obtained and returned

ArenaMemoryResource:
    one particular allocation strategy
```

## Allocation Responsibilities

`Allocation` should contain enough information to represent and correctly release one memory block:

```text
pointer
byte capacity
alignment
device
originating memory resource
optional allocation ID
```

It should provide:

- Move-only ownership
- Automatic non-throwing cleanup
- Null-safe empty state
- Pointer access
- Byte-capacity access
- Device access
- Alignment information
- An explicit reset operation if needed

Core invariants:

```text
empty allocation:
    pointer = null
    capacity = 0

nonempty allocation:
    pointer != null
    capacity > 0
    resource is valid
    device describes the pointer's memory location
```

A successfully returned `Allocation` should not contain an error state. Allocation errors should be reported by the resource before an invalid object escapes.

## What Allocation Should Not Do

It should not contain:

- Shape
- Strides
- Dtype
- Tensor operations
- Copies or transfers
- Resizing policy
- Pooling logic
- Arena bookkeeping
- CUDA kernel behavior

It owns bytes; it does not interpret them.

## Resource Relationship

Conceptually:

```text
resource.allocate(request)
    obtains memory
    returns Allocation tied to that resource

Allocation destruction
    returns memory to its originating resource
```

The allocation request can include:

```text
byte count
alignment
device
stream
usage category
```

The resource must outlive its allocations. You can guarantee this through process-lifetime resources or shared resource ownership.

## Initial Resources

Start with direct resources:

```text
HostMemoryResource:
    aligned host allocation
    corresponding host deallocation

CudaMemoryResource:
    cudaMalloc
    cudaFree
```

These establish correctness and provide a baseline for later performance comparisons.

## Arena Resource

An arena owns a larger backing region and serves smaller suballocations:

```text
ArenaMemoryResource:
    acquire one large block
    divide it into aligned regions
    track free regions
    merge released regions
```

Allocations returned by the arena still look like ordinary `Allocation` objects. Their cleanup returns the region to the arena rather than directly freeing the backing block.

An arena introduces substantial policy:

- Initial arena capacity
- Growth strategy
- Alignment
- Fragmentation
- Free-list organization
- Block coalescing
- Thread safety
- Resource lifetime
- Out-of-memory behavior
- Dedicated handling for oversized allocations

A CUDA arena additionally needs:

- Device association
- Stream safety
- CUDA events
- Deferred block reuse
- Graph-capture compatibility
- Cross-stream synchronization

Reusing a CUDA region before work on its previous stream completes can corrupt data. Therefore, a naive CUDA arena is not production-safe.

## Recommended Direction

```text
Allocation
    general ownership object

MemoryResource
    general allocation-policy interface

HostMemoryResource
    direct baseline

CudaMemoryResource
    direct baseline

ArenaMemoryResource
    later optimization

CudaPoolMemoryResource
    later stream-aware optimization
```

Do not name the general interface `ArenaAllocator`, because that commits every allocation to one strategy. Keep `Allocation` general and add an arena behind `MemoryResource` only after profiling shows direct allocation is a problem.

For Bumblebee, the distinctive scalable design is that `ExecutionContext` selects the resource:

```text
ExecutionContext:
    device
    stream
    selected MemoryResource
```

Tensor operations allocate through their context, while `Allocation` remains unaware of whether its memory came from direct allocation, an arena, or a CUDA pool.
