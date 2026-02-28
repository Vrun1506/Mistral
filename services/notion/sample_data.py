"""
Sample output from the clustering pipeline (Step 7).

Structure:
  Topic → Subtopics → Segments → Turns

Used by main.py to test the full flow without the real pipeline.
"""

SAMPLE_TOPIC_HIERARCHY = [
    {
        "cluster_id": 1,
        "topic_label": "Memory Management in C++",
        "subtopics": [
            {
                "subtopic_label": "Stack vs Heap Allocation",
                "segments": [
                    {
                        "segment_id": "seg-001",
                        "convo_id": "convo-abc",
                        "turns": [
                            {
                                "role": "user",
                                "content": "What's the difference between stack and heap allocation in C++?",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "Stack allocation is automatic and fast — memory is allocated "
                                    "when a function is called and freed when it returns. Heap "
                                    "allocation uses new/delete and persists until explicitly freed. "
                                    "Stack is limited in size (typically 1-8 MB), while heap can "
                                    "grow to available system memory."
                                ),
                            },
                        ],
                    },
                    {
                        "segment_id": "seg-002",
                        "convo_id": "convo-def",
                        "turns": [
                            {
                                "role": "user",
                                "content": "When should I use heap vs stack in a game engine?",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "In game engines, prefer stack for small short-lived objects "
                                    "like math vectors. Use heap for large objects like textures "
                                    "and meshes that outlive a single frame. Many engines use "
                                    "custom allocators (pool, arena) on top of heap."
                                ),
                            },
                        ],
                    },
                ],
            },
            {
                "subtopic_label": "Smart Pointers and RAII",
                "segments": [
                    {
                        "segment_id": "seg-003",
                        "convo_id": "convo-abc",
                        "turns": [
                            {
                                "role": "user",
                                "content": "Explain RAII and smart pointers in modern C++.",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "RAII ties resource lifetime to object scope. Smart pointers "
                                    "(unique_ptr, shared_ptr, weak_ptr) apply RAII to heap memory. "
                                    "unique_ptr has exclusive ownership and zero overhead. "
                                    "shared_ptr uses reference counting for shared ownership."
                                ),
                            },
                        ],
                    },
                ],
            },
        ],
    },
    {
        "cluster_id": 2,
        "topic_label": "NumPy Broadcasting and Linear Algebra",
        "subtopics": [
            {
                "subtopic_label": "Broadcasting Rules",
                "segments": [
                    {
                        "segment_id": "seg-004",
                        "convo_id": "convo-ghi",
                        "turns": [
                            {
                                "role": "user",
                                "content": "How does NumPy broadcasting work?",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "Broadcasting follows three rules: (1) If arrays differ in "
                                    "ndim, the smaller shape is padded with 1s on the left. "
                                    "(2) Size-1 dimensions stretch to match. (3) Arrays are "
                                    "incompatible if sizes differ and neither is 1."
                                ),
                            },
                        ],
                    },
                ],
            },
            {
                "subtopic_label": "Matrix Operations",
                "segments": [
                    {
                        "segment_id": "seg-005",
                        "convo_id": "convo-ghi",
                        "turns": [
                            {
                                "role": "user",
                                "content": "Difference between np.dot, np.matmul, and @?",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "np.dot is the general dot product but has confusing "
                                    "behavior for 3D+ arrays. np.matmul (and @) is strictly "
                                    "matrix multiplication and handles batched operations "
                                    "correctly. For 2D arrays they are identical. Prefer @."
                                ),
                            },
                        ],
                    },
                ],
            },
        ],
    },
]
