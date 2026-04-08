import json
from typing import Dict, Any, Optional


class TOCProcessor:
    def __init__(self, toc_file: str = "toc.json"):
        with open(toc_file, "r", encoding="utf-8") as f:
            self.toc = json.load(f)

        self.report_title = self.toc.get("report_title", "Unknown Report")
        self.total_pages = self.toc.get("total_pages", 0)
        self.structure = self.toc.get("structure", [])

        self.page_metadata: Dict[int, Dict[str, Any]] = {}
        self._build_page_metadata()

    def _build_page_metadata(self):
        for part in self.structure:
            part_title = part.get("part") or part.get("title", "")

            for chapter in part.get("chapters", []):
                chapter_num = chapter.get("chapter")
                chapter_title = chapter.get("title", "")

                for section in chapter.get("sections", []):
                    page = section.get("page")
                    if page:
                        self.page_metadata[page] = {
                            "report_title": self.report_title,
                            "part": part_title,
                            "chapter": chapter_num,
                            "chapter_title": chapter_title,
                            "section_id": section.get("id"),
                            "section_title": section.get("title"),
                        }

                # Process districts if any
                for district in chapter.get("districts", []):
                    page = district.get("page")
                    if page:
                        self.page_metadata[page] = {
                            "report_title": self.report_title,
                            "part": part_title,
                            "chapter": chapter_num,
                            "chapter_title": chapter_title,
                            "section_id": district.get("id"),
                            "section_title": district.get("name"),
                        }

                chapter_page = chapter.get("page")
                if chapter_page and chapter_page not in self.page_metadata:
                    self.page_metadata[chapter_page] = {
                        "report_title": self.report_title,
                        "part": part_title,
                        "chapter": chapter_num,
                        "chapter_title": chapter_title,
                    }

    def get_metadata_for_page(self, page_no: Optional[int]) -> Dict[str, Any]:
        if not page_no or page_no < 1:
            return {"report_title": self.report_title}

        if page_no in self.page_metadata:
            return self.page_metadata[page_no]

        for p in range(page_no, 0, -1):
            if p in self.page_metadata:
                meta = self.page_metadata[p].copy()
                meta["note"] = f"metadata_from_closest_page_{p}"
                return meta

        return {"report_title": self.report_title}
