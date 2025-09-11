"""
Example illustrating the BPE merge validation bug and fix.

The bug occurs when stale entries in the priority queue are processed
because they appear "active" but actually represent different token pairs
than when they were originally added.
"""

# Simplified example of what happens with the word "Hello"
# Assume these are the relevant merges in order of priority:
merges = [
    (b'l', b'l', 37),      # Merge 'l' + 'l' -> 'll'
    (b'e', b'l', 160),     # Merge 'e' + 'l' -> 'el'  
    (b'e', b'll', 280),    # Merge 'e' + 'll' -> 'ell'
    (b'H', b'e', 770),     # Merge 'H' + 'e' -> 'He'
    (b'ell', b'o', 923),   # Merge 'ell' + 'o' -> 'ello'
    (b'H', b'ello', 925),  # Merge 'H' + 'ello' -> 'Hello'
]

print("=" * 60)
print("ORIGINAL BUGGY BEHAVIOR")
print("=" * 60)

# Initial state: H -> e -> l -> l -> o
# Priority queue initially contains:
initial_pq = [
    (37, "node_l1"),   # l + l (nodes at positions 2,3)
    (160, "node_e"),   # e + l (nodes at positions 1,2)
    (770, "node_H"),   # H + e (nodes at positions 0,1)
]

print("\nInitial tokens: ['H', 'e', 'l', 'l', 'o']")
print("Initial priority queue:")
for priority, node_ref in sorted(initial_pq):
    print(f"  Priority {priority}: {node_ref}")

print("\n--- Processing merges ---")

# Step 1: Pop priority 37 (l + l)
print("\n1. Pop priority 37: Merge 'l' + 'l' -> 'll'")
print("   Tokens become: ['H', 'e', 'll', 'o']")
print("   Add to queue: (280, 'e' + 'll')")
print("   Mark old 'l' nodes as inactive")

# But here's the problem: The entry (160, node_e) is still in the queue!
# node_e originally pointed to 'e' + 'l', but now:
# - The first 'l' node is marked inactive (merged into 'll')
# - But node_e itself (the 'e' node) is still active
# - node_e.next now points to the 'll' node (which is also active)

print("\n2. Pop priority 160: This was originally 'e' + 'l'")
print("   BUG: The old code checks:")
print("   - Is node_e active? YES (the 'e' node wasn't merged)")
print("   - Is node_e.next active? YES (it now points to 'll' node)")
print("   - So it performs the merge!")
print("   But it merges 'e' + 'll' using priority 160 (wrong!)")
print("   This should have been priority 280")

print("\n   INCORRECT: Merges 'e' + 'll' at wrong priority")
print("   Tokens become: ['H', 'ell', 'o']")

print("\n3. Later merges get disrupted because of wrong merge order")
print("   Final incorrect result: ['Hell', 'o'] instead of ['Hello']")

print("\n" + "=" * 60)
print("FIXED BEHAVIOR")
print("=" * 60)

# With the fix, we store the original pair data with each queue entry
fixed_initial_pq = [
    (37, (b'l', b'l'), "node_l1"),    # Store original pair
    (160, (b'e', b'l'), "node_e"),    # Store original pair
    (770, (b'H', b'e'), "node_H"),    # Store original pair
]

print("\nInitial tokens: ['H', 'e', 'l', 'l', 'o']")
print("Initial priority queue (now with pair data):")
for priority, pair, node_ref in sorted(fixed_initial_pq):
    print(f"  Priority {priority}: {pair} at {node_ref}")

print("\n--- Processing merges ---")

print("\n1. Pop priority 37: Merge 'l' + 'l' -> 'll'")
print("   Tokens become: ['H', 'e', 'll', 'o']")
print("   Add to queue: (280, ('e', 'll'), node_e)")

print("\n2. Pop priority 160: Original pair was ('e', 'l')")
print("   FIX: The new code checks:")
print("   - Is node_e active? YES")
print("   - Is node_e.next active? YES")
print("   - Does node_e.data == 'e'? YES")
print("   - Does node_e.next.data == 'l'? NO! (it's now 'll')")
print("   - SKIP this stale entry!")

print("\n3. Pop priority 280: Merge 'e' + 'll' -> 'ell'")
print("   Tokens become: ['H', 'ell', 'o']")
print("   Add to queue: (923, ('ell', 'o'), node_ell)")

print("\n4. Pop priority 770: Original pair was ('H', 'e')")
print("   Check: node_H.data='H', node_H.next.data='ell' (not 'e')")
print("   SKIP this stale entry!")

print("\n5. Pop priority 923: Merge 'ell' + 'o' -> 'ello'")
print("   Tokens become: ['H', 'ello']")
print("   Add to queue: (925, ('H', 'ello'), node_H)")

print("\n6. Pop priority 925: Merge 'H' + 'ello' -> 'Hello'")
print("   Tokens become: ['Hello']")

print("\n   CORRECT: Final result is ['Hello'] as expected!")

print("\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("""
The bug happened because when nodes are merged, references in the
priority queue become stale. A node reference might still point to
an active node, but that node's neighbors have changed due to merges.

The fix adds the original token pair data to each queue entry and
validates that the nodes still contain the same data before merging.
This ensures we only process merges that are still valid.
""")