"""
Explanation of the counter in the BPE algorithm.

The counter serves as a tie-breaker when two merges have the same priority.
It's a monotonically increasing integer that ensures a deterministic order.
"""

import heapq

print("=" * 70)
print("WHY WE NEED THE COUNTER")
print("=" * 70)

print("\nPython's heapq compares tuples element by element:")
print("  (priority, ...) is compared first by priority")
print("  If priorities are equal, it compares the next element")
print("  If that's a _Node object, we get: TypeError: '<' not supported")

print("\n" + "=" * 70)
print("WHAT THE COUNTER REPRESENTS")
print("=" * 70)

print("""
The counter is a SEQUENCE NUMBER that increases each time we add to the queue:
  - counter = 0  → First merge added
  - counter = 1  → Second merge added
  - counter = 2  → Third merge added
  - etc.

It represents the ORDER in which merges were discovered/added to the queue.
""")

print("=" * 70)
print("CAN COUNTERS BE EQUAL? NO!")
print("=" * 70)

print("""
No, counters can NEVER be equal because:

1. We start with counter = 0
2. Every time we add to the queue: counter += 1
3. Each merge gets a UNIQUE counter value

Example sequence:
""")

# Simulate the counter usage
counter = 0
merges_added = []

# Initial merges
initial_merges = [
    ("l", "l", 37),
    ("e", "l", 160),
    ("e", "ll", 160),  # Same priority as previous!
    ("H", "e", 770),
]

print("Initial queue population:")
for left, right, priority in initial_merges:
    merges_added.append((priority, counter, f"{left}+{right}"))
    print(f"  Add ({priority}, {counter}, '{left}+{right}')")
    counter += 1

print("\nAfter processing a merge, adding new candidates:")
new_merges = [
    ("ell", "o", 923),
    ("H", "ello", 925),
]

for left, right, priority in new_merges:
    merges_added.append((priority, counter, f"{left}+{right}"))
    print(f"  Add ({priority}, {counter}, '{left}+{right}')")
    counter += 1

print("\n" + "=" * 70)
print("HOW HEAP COMPARISON WORKS")
print("=" * 70)

print("\nWhen two items have the same priority:")
item1 = (160, 1, "data1", "node1")
item2 = (160, 2, "data2", "node2")

print(f"  Item 1: {item1[:3]}...")
print(f"  Item 2: {item2[:3]}...")
print(f"\nComparison: (160, 1, ...) < (160, 2, ...)")
print(f"  1. Compare priority: 160 == 160 (equal, continue)")
print(f"  2. Compare counter: 1 < 2 (Item 1 comes first!)")
print(f"  3. Never need to compare the _Node objects")

print("\n" + "=" * 70)
print("DETERMINISTIC ORDERING")
print("=" * 70)

print("""
The counter ensures DETERMINISTIC ordering even with equal priorities:

Priority  Counter  Merge       Order in heap
--------  -------  ----------  -------------
160       1        'e'+'l'     1st (added earlier)
160       2        'e'+'ll'    2nd (added later)
160       8        'e'+'n'     3rd (added even later)

This preserves the "first-in-first-out" property for equal priorities,
which makes the algorithm behavior predictable and debuggable.
""")

print("\n" + "=" * 70)
print("ALTERNATIVE SOLUTIONS (less elegant)")
print("=" * 70)

print("""
Without the counter, we could:

1. Use random tie-breaker: 
   (priority, random.random(), ...)
   Problem: Non-deterministic, hard to debug

2. Use id() of nodes:
   (priority, id(node), ...)  
   Problem: id() values are memory addresses, non-deterministic across runs

3. Make _Node comparable:
   Add __lt__ method to _Node class
   Problem: More complex, what comparison logic to use?

4. Use a wrapper class:
   @dataclass(order=True) with priority field
   Problem: More code, more complex

The counter is the simplest, most elegant solution!
""")