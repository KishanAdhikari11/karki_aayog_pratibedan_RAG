import json
from typing import Any, Dict, Optional


class TOCProcessor:
    def __init__(self, toc_file: str = "toc.json"):
        with open(toc_file, "r", encoding="utf-8") as f:
            self.toc = json.load(f)

        self.report_title: str = self.toc.get("report_title", "Unknown Report")
        self.total_pages: int = self.toc.get("total_pages", 0)
        self.structure: list = self.toc.get("structure", [])

        self.page_metadata: Dict[int, Dict[str, Any]] = {}
        self._build_page_metadata()

    def _register(self, page: Optional[int], meta: Dict[str, Any]):
        """Register metadata for a page if not already registered."""
        if page and page not in self.page_metadata:
            self.page_metadata[page] = {
                "report_title": self.report_title,
                **meta,
            }

    def _build_page_metadata(self):
        for part in self.structure:
            part_title = part.get("part") or part.get("title", "")

            for chapter in part.get("chapters", []):
                chapter_num = chapter.get("chapter")
                chapter_title = chapter.get("title", "")

                base_meta = {
                    "part": part_title,
                    "chapter": chapter_num,
                    "chapter_title": chapter_title,
                }

                self._register(chapter.get("page"), base_meta)

                for section in chapter.get("sections", []):
                    self._register(
                        section.get("page"),
                        {
                            **base_meta,
                            "section_id": section.get("id"),
                            "section_title": section.get("title"),
                        },
                    )

                for district in chapter.get("districts", []):
                    self._register(
                        district.get("page"),
                        {
                            **base_meta,
                            "section_id": district.get("id"),
                            "section_title": district.get("name"),
                        },
                    )

    def get_metadata_for_page(self, page_no: Optional[int]) -> Dict[str, Any]:
        """
        Return metadata for a given page number.
        If the exact page isn't in the TOC, walk backwards to find the
        nearest preceding registered page (i.e. the section this page belongs to).
        """
        if not page_no or page_no < 1:
            return {"report_title": self.report_title}

        if page_no in self.page_metadata:
            return self.page_metadata[page_no]

        for p in range(page_no - 1, 0, -1):
            if p in self.page_metadata:
                return self.page_metadata[p]

        return {"report_title": self.report_title}
