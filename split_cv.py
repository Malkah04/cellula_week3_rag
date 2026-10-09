from langchain_core.documents import Document
import re

SECTION_HEADINGS = [
    "PROFESSIONAL SUMMARY",
    "EDUCATION",
    "CERTIFICATIONS",
    "SKILLS",
    "PROJECTS",
    "EXPERIENCE",
    "COMPETITIONS & ACTIVITIES",
]

PROJECT_TITLES = [
    "Adversarial Attacks & Defenses",
    "Data Poisoning Attacks & Defenses",
    "SecureSentinel AI API",
    "Federated Heart Disease Classification",
    "Ironwall - Privacy-Preserving Network Security",
]

JOB_TITLES = [
    "Machine Learning & Data Analysis Trainee | NTI",
    "MEAN Stack Intern | NTI",
]


def split_named_items(text, titles, section_name, source_file):
    pattern = re.compile(
        "|".join(re.escape(title) for title in titles),
        re.IGNORECASE
    )

    matches = list(pattern.finditer(text))
    documents = []

    for i, match in enumerate(matches):
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        documents.append(
            Document(
                page_content=f"{section_name}\n{content}",
                metadata={
                    "category": "CV",
                    "section": section_name,
                    "item": match.group().strip(),
                    "source_file": source_file,
                },
            )
        )

    return documents


def split_cv_sections(pages):
    full_text = "\n".join(
        page.page_content for page in pages
    )

    source_file = pages[0].metadata.get("source_file","CV")

    heading_pattern = re.compile(
        r"(?m)^(?:"
        + "|".join(
            re.escape(h) for h in SECTION_HEADINGS
        )
        + r")\s*$",
        re.IGNORECASE,
    )

    headings = list(heading_pattern.finditer(full_text))
    documents = []

    for i, heading in enumerate(headings):
        section_name = heading.group().strip()

        if i + 1 < len(headings):
            section_end = headings[i + 1].start()
        else:
            section_end = len(full_text)

        section_text = full_text[
            heading.end():section_end
        ].strip()

        if not section_text:
            continue

        if section_name.upper() == "PROJECTS":
            documents.extend(
                split_named_items(
                    section_text,
                    PROJECT_TITLES,
                    "Projects",
                    source_file,
                )
            )

        elif section_name.upper() == "EXPERIENCE":
            documents.extend(
                split_named_items(
                    section_text,
                    JOB_TITLES,
                    "Experience",
                    source_file,
                )
            )

        else:
            documents.append(
                Document(
                    page_content=(
                        f"{section_name}\n{section_text}"
                    ),
                    metadata={
                        "category": "CV",
                        "section": section_name,
                        "source_file": source_file,
                    },
                )
            )

    return documents
