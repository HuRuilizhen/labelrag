"""Coverage helpers for greedy label-based retrieval."""


def uncovered_overlap_size(
    paragraph_label_ids: set[str],
    remaining_label_ids: set[str],
) -> int:
    """Return the count of uncovered query labels matched by a paragraph."""

    return len(paragraph_label_ids & remaining_label_ids)
