# convoviz/models/node.py
# GPT-5.6 Sol | CONVOVIZ-FORK-FIDELITY-FIX-20260913 | 2026-09-14
"""Node model - pure data class.

Object path: conversations.json -> conversation -> mapping -> mapping node

Nodes form a tree structure representing conversation branches.
"""

from pydantic import BaseModel, Field

from convoviz.models.message import Message


class Node(BaseModel):
    """A node in the conversation tree.

    Each node can have a message and links to parent/children nodes.
    This is a pure data model - rendering logic is in the renderers module.
    """

    id: str
    message: Message | None = None
    parent: str | None = None
    children: list[str] = Field(default_factory=list)

    # Runtime-populated references (not from JSON)
    parent_node: "Node | None" = None
    children_nodes: list["Node"] = Field(default_factory=list)

    def add_child(self, node: "Node") -> None:
        """Add a child node and set up bidirectional references."""
        self.children_nodes.append(node)
        node.parent_node = self

    @property
    def has_message(self) -> bool:
        """Check if this node contains a message."""
        return self.message is not None


def node_sort_key(node: Node) -> tuple[float, str]:
    """Sort nodes by message timestamp (fallback 0), then by node ID."""
    message = node.message
    if message and message.create_time:
        try:
            return (message.create_time.timestamp(), node.id)
        except Exception:
            pass
    return (0.0, node.id)


def build_node_tree(mapping: dict[str, Node]) -> dict[str, Node]:
    """Build the node tree by connecting parent/child references.

    Args:
        mapping: Dictionary of node_id -> Node

    Returns:
        The same dictionary with nodes connected via parent_node/children_nodes

    """
    # Reset connections to avoid duplicates on repeated calls
    for node in mapping.values():
        node.children_nodes = []
        node.parent_node = None

    # Parent pointers are authoritative in current ChatGPT exports.
    # Newer exports may omit `children` entirely, while older exports include both.
    for node in mapping.values():
        if node.parent and node.parent in mapping:
            mapping[node.parent].add_child(node)

    # Backward compatibility for child-only/legacy records: use explicit child lists
    # only when the child has no usable parent pointer. If both disagree, trust parent.
    for node in mapping.values():
        for child_id in node.children:
            if child_id not in mapping:
                continue
            child_node = mapping[child_id]
            if child_node.parent_node is None and not (
                child_node.parent and child_node.parent in mapping
            ):
                node.add_child(child_node)

    return mapping
