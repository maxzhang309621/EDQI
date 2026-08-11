"""S3 图纸接入：图像/PDF -> 页图 + meta。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline import resolve_path


def _drawing_id_from_path(path: Path) -> str:
    return path.stem


def _scale_image(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / float(longest)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.LANCZOS), scale


def _ingest_sig(src: Path, *, max_side: int, dpi: int, page: int) -> str:
    st = src.stat()
    return f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}|{max_side}|{dpi}|{page}"


def ingest(
    drawing_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    drawing_id: str | None = None,
    max_side: int = 2048,
    dpi: int = 200,
    page: int = 1,
) -> tuple[Path, dict[str, Any]]:
    """
    返回 (page_image_path, meta)。
    meta 含 width/height/page/dpi/scale_to_vlm/source_path，供 bbox 还原。

    同源同参数时复用已生成页图，避免改写 mtime 导致感知缓存永远 miss。
    """
    src = resolve_path(drawing_path)
    if not src.exists():
        raise FileNotFoundError(src)

    did = drawing_id or _drawing_id_from_path(src)
    out_root = resolve_path(output_dir or "work_dirs/ingest") / did
    out_root.mkdir(parents=True, exist_ok=True)

    suffix = src.suffix.lower()
    page_path = out_root / f"page_{page:02d}.png"
    meta_path = out_root / f"page_{page:02d}_meta.json"
    sig = _ingest_sig(src, max_side=max_side, dpi=dpi, page=page)

    if meta_path.exists() and page_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict) and old.get("ingest_sig") == sig:
                old.setdefault("page_image", str(page_path))
                old.setdefault("source_path", str(src))
                return page_path, old
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError("处理 PDF 需要安装 pymupdf") from e
        doc = fitz.open(src)
        if page < 1 or page > len(doc):
            raise ValueError(f"页码越界: {page}/{len(doc)}")
        p = doc[page - 1]
        zoom = dpi / 72.0
        pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        raw_path = out_root / f"page_{page:02d}_raw.png"
        pix.save(str(raw_path))
        img = Image.open(raw_path).convert("RGB")
        doc.close()
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        img = Image.open(src).convert("RGB")
        page = 1
        page_path = out_root / f"page_{page:02d}.png"
        meta_path = out_root / f"page_{page:02d}_meta.json"
        sig = _ingest_sig(src, max_side=max_side, dpi=dpi, page=page)
    else:
        raise ValueError(f"暂不支持的输入格式: {suffix}（MVP 支持图像/PDF）")

    orig_w, orig_h = img.size
    scaled, scale = _scale_image(img, max_side)
    scaled.save(page_path)

    meta = {
        "drawing_id": did,
        "width": scaled.size[0],
        "height": scaled.size[1],
        "orig_width": orig_w,
        "orig_height": orig_h,
        "page": page,
        "dpi": dpi,
        "scale_to_vlm": scale,
        "source_path": str(src),
        "page_image": str(page_path),
        "ingest_sig": sig,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return page_path, meta
