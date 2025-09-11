"""
More accurate example that actually triggers the TypeError.

The error occurs during heap operations when the heap needs to
rebalance and compare elements.
"""

import heapq
import random

print("=" * 70)
print("ACTUAL TypeError DEMONSTRATION")
print("=" * 70)

class _Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.is_active = True
    
    def __repr__(self):
        return f"Node({self.data})"

print("\nThe TypeError happens when heapq needs to rebalance the heap")
print("and encounters equal priorities. Let's force this situation:")

print("\n1. WITHOUT COUNTER (will error):")
print("-" * 40)

pq_broken = []
nodes = [_Node(bytes([i])) for i in range(10)]

# Add multiple items with the SAME priority to force comparisons
print("Adding multiple items with priority 100:")
try:
    for i, node in enumerate(nodes[:5]):
        heapq.heappush(pq_broken, (100, node))
        print(f"  Added (100, {node})")
except TypeError as e:
    print(f"\n  ERROR on item {i}! TypeError: {e}")
    print("  This happens when heap tries to compare two (100, Node) tuples")
    print("  during internal rebalancing!")

print("\nNow adding items with DIFFERENT priorities to force rebalancing:")
# This forces heap to rebalance and compare the equal-priority items
try:
    for i in range(5, 10):
        priority = random.randint(50, 150)
        heapq.heappush(pq_broken, (priority, nodes[i]))
        print(f"  Added ({priority}, {nodes[i]})")
        
    # Try to pop items - this often triggers comparison
    print("\nPopping items (this may trigger comparison):")
    while pq_broken:
        item = heapq.heappop(pq_broken)
        print(f"  Popped: {item}")
        
except TypeError as e:
    print(f"\n  ERROR! TypeError: {e}")
    print("  This happens when heap needs to compare two (100, Node) tuples")

print("\n" + "=" * 70)
print("2. WITH COUNTER (works fine):")
print("-" * 40)

pq_fixed = []
nodes = [_Node(bytes([i])) for i in range(10)]
counter = 0

print("Adding multiple items with priority 100 (with counter):")
for i, node in enumerate(nodes[:5]):
    heapq.heappush(pq_fixed, (100, counter, node))
    print(f"  Added (100, {counter}, {node})")
    counter += 1

print("\nAdding items with different priorities:")
for i in range(5, 10):
    priority = random.randint(50, 150)
    heapq.heappush(pq_fixed, (priority, counter, nodes[i]))
    print(f"  Added ({priority}, {counter}, {nodes[i]})")
    counter += 1

print("\nPopping all items (no error!):")
while pq_fixed:
    priority, cnt, node = heapq.heappop(pq_fixed)
    print(f"  Popped: priority={priority}, counter={cnt}, node={node}")

print("\n" + "=" * 70)
print("WHEN DOES THE ERROR ACTUALLY OCCUR?")
print("=" * 70)

print("""
The TypeError occurs specifically when:

1. Two heap entries have the SAME priority
2. The heap needs to compare them during operations like:
   - Inserting a new item (heappush) that causes rebalancing
   - Removing an item (heappop) that causes rebalancing
   - The heap internally reorganizing (heapify)

The comparison happens inside heapq's internal algorithms when it
needs to decide which of two equal-priority items should be higher
in the heap tree.

In the BPE tokenizer, this commonly happens because many different
token pairs can have the same merge priority (they appear equally
often in the training data).
""")