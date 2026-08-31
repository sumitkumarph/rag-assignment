from typing import List


def create_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[str]:

    if not text:
        return []

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Recursive separators
    separators = [
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]

    def split_text(
        text: str,
        separators: List[str]
    ) -> List[str]:

        if len(text) <= chunk_size:
            return [text.strip()]

        separator = separators[-1]

        for sep in separators:

            if sep == "":
                separator = sep
                break

            if sep in text:
                separator = sep
                break

        parts = text.split(separator)

        chunks = []
        current = ""

        for part in parts:

            part = part.strip()

            if not part:
                continue

            candidate = (
                current + separator + part
                if current
                else part
            )

            if len(candidate) <= chunk_size:

                current = candidate

            else:

                if current:
                    chunks.append(current.strip())

                # If individual part is too large,
                # recursively split it.
                if len(part) > chunk_size and len(separators) > 1:

                    smaller = split_text(
                        part,
                        separators[1:]
                    )

                    chunks.extend(smaller)

                    current = ""

                else:
                    current = part

        if current:
            chunks.append(current.strip())

        return chunks

    raw_chunks = split_text(
        text,
        separators
    )

    # Add overlap
    final_chunks = []

    for i, chunk in enumerate(raw_chunks):

        if i == 0:

            final_chunks.append(chunk)

        else:

            previous = raw_chunks[i - 1]

            overlap_text = previous[
                max(0, len(previous) - chunk_overlap):
            ]

            combined = (
                overlap_text + "\n" + chunk
            )

            final_chunks.append(
                combined.strip()
            )

    return final_chunks