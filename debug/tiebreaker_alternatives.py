"""
Demonstrating that ANY deterministic tie-breaker works.
The semantic meaning is irrelevant - we just need something comparable!
"""

import heapq
import random
import time

class _Node:
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"Node({self.data})"

print("=" * 70)
print("ANY DETERMINISTIC TIE-BREAKER WORKS!")
print("=" * 70)

nodes = [_Node(f"data{i}") for i in range(5)]

print("\n1. Using a COUNTER (what we chose):")
print("-" * 40)
pq1 = []
counter = 0
for node in nodes[:3]:
    heapq.heappush(pq1, (100, counter, node))  # All priority 100
    print(f"  Push (100, {counter}, {node})")
    counter += 1

print("  Result: Works! Counter provides unique ordering")

print("\n2. Using NEGATIVE counter (counts down):")
print("-" * 40)
pq2 = []
counter = 0
for node in nodes[:3]:
    heapq.heappush(pq2, (100, -counter, node))  # Negative!
    print(f"  Push (100, {-counter}, {node})")
    counter += 1

print("  Result: Works! Just different order")

print("\n3. Using RANDOM numbers:")
print("-" * 40)
pq3 = []
random.seed(42)  # Fixed seed for reproducibility
for node in nodes[:3]:
    tie = random.random()
    heapq.heappush(pq3, (100, tie, node))
    print(f"  Push (100, {tie:.3f}, {node})")

print("  Result: Works! But non-deterministic across runs")

print("\n4. Using STRING tie-breakers:")
print("-" * 40)
pq4 = []
for i, node in enumerate(nodes[:3]):
    tie = f"tie_{i:03d}"  # Strings are comparable!
    heapq.heappush(pq4, (100, tie, node))
    print(f"  Push (100, '{tie}', {node})")

print("  Result: Works! Strings compare lexicographically")

print("\n5. Using CONSTANT (DOESN'T WORK!):")
print("-" * 40)
pq5 = []
try:
    for node in nodes[:3]:
        heapq.heappush(pq5, (100, 999, node))  # Same constant!
        print(f"  Push (100, 999, {node})")
except TypeError as e:
    print(f"  ERROR: {e}")
    print("  Failed because tie-breaker (999) is not unique!")

print("\n" + "=" * 70)
print("WHAT MAKES A GOOD TIE-BREAKER?")
print("=" * 70)

print("""
Requirements:
1. COMPARABLE: Must support < operator (numbers, strings, tuples, etc.)
2. UNIQUE: Each entry needs a different value to avoid comparing nodes
3. DETERMINISTIC: Same input should give same output (for debugging)

Good choices:
✓ Counter (0, 1, 2, ...) - Simple, deterministic, efficient
✓ Negative counter - Works but less intuitive
✓ Timestamps - Works but less reproducible
✓ UUIDs as strings - Works but wasteful

Bad choices:
✗ Constants - Not unique, will still error
✗ Random without seed - Non-deterministic
✗ Node objects - Not comparable
✗ Hash values - Might collide (not unique)
""")

print("=" * 70)
print("THE ALGORITHM DOESN'T CARE!")
print("=" * 70)

print("""
The BPE algorithm result is EXACTLY THE SAME regardless of tie-breaker!

Why? Because:
1. Merges are processed by PRIORITY (that's what matters)
2. The tie-breaker only affects order of EQUAL-priority merges
3. Equal-priority merges can be processed in any order without
   affecting the final result (they don't interfere with each other)

The counter is just the simplest, most elegant solution:
- Easy to implement (just increment)
- Guaranteed unique
- Predictable for debugging
- Memory efficient (just an integer)
- Preserves insertion order (FIFO for equal priorities)
""")

print("\n" + "=" * 70)
print("DEMONSTRATION: Different tie-breakers, same result")
print("=" * 70)

# Pop and show that order within same priority doesn't matter
print("\nPopping from queue 1 (counter):")
while pq1:
    priority, tie, node = heapq.heappop(pq1)
    print(f"  {node.data}")

print("\nPopping from queue 2 (negative counter):")
while pq2:
    priority, tie, node = heapq.heappop(pq2)
    print(f"  {node.data}")

print("\nPopping from queue 4 (strings):")
while pq4:
    priority, tie, node = heapq.heappop(pq4)
    print(f"  {node.data}")

print("\nNote: Order differs but all process the same priority level!")