"""
Concrete example demonstrating the TypeError with heapq and _Node objects.

This shows exactly what happens when priorities are equal and Python
tries to compare _Node objects.
"""

import heapq

print("=" * 70)
print("DEMONSTRATION OF THE TypeError")
print("=" * 70)

# First, let's create a simple Node class like in the tokenizer
class _Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.is_active = True
    
    def __repr__(self):
        return f"Node({self.data})"

print("\n1. Creating some nodes:")
node_e = _Node(b'e')
node_l = _Node(b'l')
node_ll = _Node(b'll')
print(f"   node_e = {node_e}")
print(f"   node_l = {node_l}")
print(f"   node_ll = {node_ll}")

print("\n" + "=" * 70)
print("THE PROBLEM: When priorities are equal")
print("=" * 70)

print("\n2. Creating a priority queue WITHOUT the counter fix:")
pq_broken = []

# Add some merges with DIFFERENT priorities - this works fine
print("\n   Adding merges with different priorities:")
heapq.heappush(pq_broken, (37, node_l))    # priority 37
print(f"   heappush((37, {node_l})) - OK!")
heapq.heappush(pq_broken, (160, node_e))   # priority 160
print(f"   heappush((160, {node_e})) - OK!")

# Now try to add another merge with THE SAME priority
print("\n   Adding a merge with EQUAL priority 160:")
try:
    heapq.heappush(pq_broken, (160, node_ll))  # SAME priority 160!
    print(f"   heappush((160, {node_ll})) - OK!")
except TypeError as e:
    print(f"   heappush((160, {node_ll})) - ERROR!")
    print(f"   TypeError: {e}")

print("\n" + "=" * 70)
print("WHY THIS HAPPENS")
print("=" * 70)

print("""
When heapq needs to maintain heap order, it compares tuples:
  
  (160, node_e) vs (160, node_ll)
  
Python compares element by element:
  1. Compare 160 vs 160 → Equal, so continue to next element
  2. Compare node_e vs node_ll → ERROR! _Node objects can't be compared
  
Python tries to do: node_e < node_ll
But _Node class doesn't define __lt__ (less than) method!
""")

print("=" * 70)
print("THE FIX: Add a counter as tie-breaker")
print("=" * 70)

print("\n3. Creating a priority queue WITH the counter fix:")
pq_fixed = []
counter = 0

print("\n   Adding the same merges, now with counter:")
heapq.heappush(pq_fixed, (37, counter, node_l))
print(f"   heappush((37, {counter}, {node_l})) - OK!")
counter += 1

heapq.heappush(pq_fixed, (160, counter, node_e))
print(f"   heappush((160, {counter}, {node_e})) - OK!")
counter += 1

heapq.heappush(pq_fixed, (160, counter, node_ll))  # Same priority, different counter
print(f"   heappush((160, {counter}, {node_ll})) - OK! No error!")
counter += 1

print("\n" + "=" * 70)
print("HOW THE FIX WORKS")
print("=" * 70)

print("""
Now when heapq compares tuples with equal priority:
  
  (160, 1, node_e) vs (160, 2, node_ll)
  
Python compares:
  1. Compare 160 vs 160 → Equal, continue
  2. Compare 1 vs 2 → 1 < 2, so first item comes first
  3. Never needs to compare the _Node objects!
  
The counter (1 vs 2) provides the tie-break, preventing the TypeError.
""")

print("=" * 70)
print("REAL EXAMPLE FROM THE BPE ALGORITHM")
print("=" * 70)

print("""
In the actual tokenizer, this happens when multiple token pairs
have the same merge priority. For example:

Word: "Hello, world"
  
Possible merges:
  'e' + 'l' → priority 160
  'e' + 'll' → priority 280  
  'e' + 'n' → priority 160  (SAME as 'e' + 'l'!)
  
Without counter: TypeError when comparing nodes
With counter: Works perfectly, processes merges in discovery order
""")

print("\n" + "=" * 70)
print("TESTING THE HEAP OPERATIONS")
print("=" * 70)

print("\n4. Popping from the fixed queue to show it works:")
while pq_fixed:
    priority, cnt, node = heapq.heappop(pq_fixed)
    print(f"   Popped: priority={priority}, counter={cnt}, node={node}")

print("\nAll operations successful with the counter fix!")