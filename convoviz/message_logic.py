# convoviz/message_logic.py
# GPT-5.6 Sol | ChatGPT Export Vault Update | 2026-09-20
"""Message interpretation utilities.

Keeps complex extraction and visibility rules outside of pure data models.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from convoviz.exceptions import MessageContentError

if TYPE_CHECKING:
    from convoviz.models.message import Message


def extract_message_images(message: Message) -> list[str]:
    """Extract image asset pointers from the message content."""
    image_ids: list[str] = []
    image_id_set: set[str] = set()

    if message.content.parts:
        for part in message.content.parts:
            if (
                isinstance(part, dict)
                and part.get("content_type") == "image_asset_pointer"
            ):
                pointer = part.get("asset_pointer", "")
                # Strip prefixes like "file-service://" or "sediment://"
                if pointer.startswith("file-service://"):
                    pointer = pointer[len("file-service://") :]
                elif pointer.startswith("sediment://"):
                    pointer = pointer[len("sediment://") :]

                if pointer and pointer not in image_id_set:
                    image_ids.append(pointer)
                    image_id_set.add(pointer)

    # Also check metadata.attachments for images/files that should be rendered
    if message.metadata.attachments:
        for att in message.metadata.attachments:
            if not isinstance(att, dict):
                continue
            att_id = att.get("id")
            if not att_id or att_id in image_id_set:
                continue
            if _is_image_attachment(att):
                image_ids.append(att_id)
                image_id_set.add(att_id)

    return image_ids


def extract_message_files(message: Message) -> list[tuple[str, str | None]]:
    """Extract non-image file attachments without duplicating images."""
    files: list[tuple[str, str | None]] = []
    seen = set(message.images)
    for att in message.metadata.attachments or []:
        if not isinstance(att, dict) or _is_image_attachment(att):
            continue
        att_id = att.get("id")
        if not isinstance(att_id, str) or not att_id or att_id in seen:
            continue
        name = att.get("name")
        files.append((att_id, name if isinstance(name, str) and name else None))
        seen.add(att_id)
    return files


def _render_thoughts(thoughts: list[dict[str, Any]] | None) -> str:
    """Render exported reasoning without dropping detailed thought content."""
    if not thoughts:
        return ""

    rendered: list[str] = []
    for thought in thoughts:
        if not isinstance(thought, dict):
            continue
        summary = thought.get("summary")
        content = thought.get("content")
        summary_text = summary.strip() if isinstance(summary, str) else ""
        content_text = content.strip() if isinstance(content, str) else ""

        if summary_text and content_text and summary_text != content_text:
            rendered.append(f"**{summary_text}**\n\n{content_text}")
        elif content_text:
            rendered.append(content_text)
        elif summary_text:
            rendered.append(summary_text)

    return "\n\n".join(rendered)


def _render_tether_quote(content: Any) -> str:
    """Render tether_quote content as a blockquote."""
    quote_text = content.text or ""
    if not quote_text.strip():
        return ""
    # Format as blockquote with source
    lines = [f"> {line}" for line in quote_text.strip().split("\n")]
    blockquote = "\n".join(lines)
    # Add attribution if we have title/domain/url
    if content.title and content.url:
        blockquote += f"\n> — [{content.title}]({content.url})"
    elif content.domain and content.url:
        blockquote += f"\n> — [{content.domain}]({content.url})"
    elif content.url:
        blockquote += f"\n> — <{content.url}>"
    return blockquote


def _render_canvas(name: str, content: str) -> str:
    """Format a Canvas/canmore document as markdown."""
    return f"### Canvas: {name}\n\n{content}"


def _parse_canvas_payload(value: Any) -> dict[str, Any] | None:
    """Parse a Canvas document from dict or JSON string."""
    if isinstance(value, dict):
        data = value
    elif isinstance(value, str):
        try:
            data = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    else:
        return None

    if not isinstance(data, dict):
        return None

    name = data.get("name")
    content = data.get("content")
    if not isinstance(name, str) or not isinstance(content, str):
        return None

    normalized: dict[str, Any] = {
        "name": name,
        "content": content,
    }
    doc_type = data.get("type")
    if isinstance(doc_type, str):
        normalized["type"] = doc_type
    return normalized


def extract_message_text(message: Message) -> str:
    """Extract the text content of the message."""
    content = message.content

    # 1. Handle multimodal parts
    if content.parts is not None:
        text_parts = []
        for part in content.parts:
            if isinstance(part, str):
                if message.recipient == "canmore.create_textdoc":
                    doc = _parse_canvas_payload(part)
                    if doc:
                        text_parts.append(_render_canvas(doc["name"], doc["content"]))
                        continue
                text_parts.append(part)
                continue

            if isinstance(part, dict):
                if message.recipient == "canmore.create_textdoc":
                    doc = _parse_canvas_payload(part)
                    if (
                        doc
                        and isinstance(doc.get("name"), str)
                        and isinstance(doc.get("content"), str)
                    ):
                        text_parts.append(_render_canvas(doc["name"], doc["content"]))
                        continue
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

        if text_parts:
            return "".join(text_parts)
        if content.parts:  # If we had non-text parts (like images)
            return ""

    # 2. Handle specific content types via match
    match content.content_type:
        case "tether_quote":
            return _render_tether_quote(content)
        case "reasoning_recap" if content.content is not None:
            return content.content
        case "thoughts" if content.thoughts is not None:
            return _render_thoughts(content.thoughts)

    # 3. Fallback to standard text/result fields
    if content.text is not None:
        if message.recipient == "canmore.create_textdoc":
            doc = _parse_canvas_payload(content.text)
            if doc:
                return _render_canvas(doc["name"], doc["content"])
        return content.text

    if content.result is not None:
        return content.result

    if extract_message_images(message) or extract_message_files(message):
        # Image-only/tool-only messages should render as image blocks,
        # not fail text extraction.
        return ""

    raise MessageContentError(message.id)


_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".svg",
}


def _is_image_attachment(att: dict[str, Any]) -> bool:
    """Return True if attachment metadata looks like an image."""
    mime = att.get("mime_type") or att.get("content_type") or att.get("file_type")
    if isinstance(mime, str) and mime.lower().startswith("image/"):
        return True
    name = att.get("name")
    if isinstance(name, str):
        lower = name.lower()
        return any(lower.endswith(ext) for ext in _IMAGE_EXTS)
    return False


def extract_canvas_document(message: Message) -> dict[str, Any] | None:
    """Extract Canvas document if this message created one."""
    if message.recipient != "canmore.create_textdoc":
        return None

    def to_doc(val: Any) -> dict[str, Any] | None:
        data = _parse_canvas_payload(val)
        if not data:
            return None
        doc_type = data.get("type")
        return {
            "name": data["name"],
            "type": doc_type if isinstance(doc_type, str) else "unknown",
            "content": data["content"],
        }

    # Try all parts in order
    if message.content.parts:
        for part in message.content.parts:
            doc = to_doc(part)
            if doc:
                return doc

    # Try content.text
    if message.content.text:
        doc = to_doc(message.content.text)
        if doc:
            return doc

    return None


def is_message_hidden(message: Message) -> bool:
    """Check if message should be hidden in export."""
    if message.is_empty or message.metadata.is_visually_hidden_from_conversation:
        return True

    match message.author.role:
        case "system":
            return not message.metadata.is_user_system_message

        case "tool":
            match message.author.name:
                case "bio" | "web.run" | "web.search":
                    return True
                case "browser":
                    return message.content.content_type != "tether_quote"
                case "dalle.text2im" if (
                    message.content.content_type == "text" and not message.images
                ):
                    return True

        case "assistant":
            # Hide code interpreter input and internal tool calls
            if message.content.content_type == "code":
                return True
            if message.recipient not in ("all", "python", None):
                return True

    # Generic content type filters
    return message.content.content_type in (
        "sonic_webpage",
        "system_error",
        "tether_browsing_display",
        "thoughts",
        "reasoning_recap",
    )


def extract_internal_citation_map(message: Message) -> dict[str, dict[str, str | None]]:
    """Extract only unambiguous embedded citation IDs for one message.

    Current exports can reuse the same turn/ref token for different sources, even
    inside one message. Conflicting primary definitions are therefore omitted so
    the renderer preserves the raw citation marker instead of misattributing it.
    Fallback metadata may fill an otherwise unknown key but never overrides a
    primary definition, regardless of metadata ordering.
    """
    primary_candidates: dict[str, dict[str, str | None]] = {}
    fallback_candidates: dict[str, dict[str, str | None]] = {}
    ambiguous_primary: set[str] = set()
    ambiguous_fallback: set[str] = set()
    parts = message.content.parts or []

    def ref_key(ref_id: Any) -> str | None:
        if not isinstance(ref_id, dict):
            return None
        ref_type = ref_id.get("ref_type")
        turn_idx = ref_id.get("turn_index")
        ref_idx = ref_id.get("ref_index")
        if not isinstance(ref_type, str) or not ref_type:
            return None
        if not isinstance(turn_idx, int) or not isinstance(ref_idx, int):
            return None
        return f"turn{turn_idx}{ref_type}{ref_idx}"

    def add_candidate(
        key: str,
        metadata: dict[str, str | None],
        *,
        fallback: bool = False,
    ) -> None:
        target = fallback_candidates if fallback else primary_candidates
        ambiguous = ambiguous_fallback if fallback else ambiguous_primary
        existing = target.get(key)
        if existing is None:
            target[key] = metadata
        elif existing != metadata:
            ambiguous.add(key)

    def process_entry(entry: Any) -> None:
        if not isinstance(entry, dict):
            return
        key = ref_key(entry.get("ref_id"))
        if key:
            add_candidate(
                key,
                {
                    "title": entry.get("title"),
                    "url": entry.get("url"),
                },
            )

    for part in parts:
        if isinstance(part, dict):
            if part.get("type") == "search_result":
                process_entry(part)
            elif part.get("type") == "search_result_group":
                for entry in part.get("entries", []):
                    process_entry(entry)

    for group in message.metadata.search_result_groups or []:
        if isinstance(group, dict):
            for entry in group.get("entries", []):
                process_entry(entry)

    for reference in message.metadata.content_references or []:
        if (
            not isinstance(reference, dict)
            or reference.get("type") != "grouped_webpages"
        ):
            continue
        for item in reference.get("items") or []:
            if not isinstance(item, dict):
                continue
            metadata = {"title": item.get("title"), "url": item.get("url")}
            for ref_id in item.get("refs") or []:
                if key := ref_key(ref_id):
                    add_candidate(key, metadata)
        for item in reference.get("fallback_items") or []:
            if not isinstance(item, dict):
                continue
            metadata = {"title": item.get("title"), "url": item.get("url")}
            for ref_id in item.get("refs") or []:
                if key := ref_key(ref_id):
                    add_candidate(key, metadata, fallback=True)

    result = {
        key: metadata
        for key, metadata in primary_candidates.items()
        if key not in ambiguous_primary
    }
    for key, metadata in fallback_candidates.items():
        if (
            key not in primary_candidates
            and key not in ambiguous_fallback
            and key not in result
        ):
            result[key] = metadata
    return result
