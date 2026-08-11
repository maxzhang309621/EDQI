"""确定性规则引擎：Facts + Rules.logic -> Findings。"""
from __future__ import annotations

import re
from typing import Any


class RuleEngineError(ValueError):
    pass


def get_by_path(facts: dict[str, Any], path: str) -> Any:
    """支持 a.b / a[0].b；列表取字段时返回该字段值列表。"""
    if not path:
        return None
    cur: Any = facts
    for part in path.split("."):
        if cur is None:
            return None
        m = re.fullmatch(r"(\w+)\[(\d+)\]", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        if isinstance(cur, list):
            # components.value -> [c['value'] for c in components]
            return [item.get(part) if isinstance(item, dict) else None for item in cur]
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True
        # components.id -> [..]：任一非空即视为有值；全空才 empty
        return all(_is_empty(v) for v in value)
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    inter = _bbox_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _count_bbox_overlaps(
    items: list[Any],
    *,
    iou_thr: float = 0.0,
    min_intersection: float = 1.0,
    min_overlap_ratio: float = 0.0,
    min_box_side: float = 0.0,
    ignore_horizontal_neighbors: bool = False,
    neighbor_y_tol: float = 0.45,
    only_keep_pair: bool = False,
) -> tuple[int, list[list[float]], list[str]]:
    """返回 (重叠对数, 涉及的框, 相关 id)。

    min_overlap_ratio: inter / min(area_a, area_b)，抑制小符号「蹭边」误报。
    ignore_horizontal_neighbors: 同行左右相邻字符（正常多位数）不记重叠。
    only_keep_pair: 仅统计感知侧已确认的重叠对（如墨迹重叠 keep_pair）。
    """
    boxes: list[tuple[list[float], str, Any]] = []
    for i, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        if only_keep_pair and not item.get("keep_pair"):
            continue
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        if min_box_side > 0 and min(w, h) < min_box_side:
            continue
        rid = str(item.get("instance_id") or item.get("id") or f"#{i}")
        boxes.append((list(bbox), rid, item))

    def _is_h_neighbor(a: list[float], b: list[float]) -> bool:
        aw, ah = a[2] - a[0], a[3] - a[1]
        bw, bh = b[2] - b[0], b[3] - b[1]
        cy_a = (a[1] + a[3]) / 2.0
        cy_b = (b[1] + b[3]) / 2.0
        avg_h = max(1.0, (ah + bh) / 2.0)
        if abs(cy_a - cy_b) > neighbor_y_tol * avg_h:
            return False
        # 左右排列：x 区间几乎不叠或只轻微接触，中心水平分开
        x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        if y_overlap < 0.5 * min(ah, bh):
            return False
        cx_a = (a[0] + a[2]) / 2.0
        cx_b = (b[0] + b[2]) / 2.0
        # 中心距接近「一字宽」，且相交面积相对小 → 相邻字
        dist = abs(cx_a - cx_b)
        char_w = max(aw, bw, 1.0)
        if dist < 0.55 * char_w:
            return False  # 太近，更像真重叠
        if dist > 2.2 * char_w:
            return False
        min_area = min(aw * ah, bw * bh)
        inter = x_overlap * y_overlap
        if min_area <= 0:
            return False
        return (inter / min_area) < 0.35

    pairs = 0
    hit_boxes: list[list[float]] = []
    hit_ids: list[str] = []
    seen_box = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ba, ida, ia = boxes[i]
            bb, idb, ib = boxes[j]
            # 同一 OCR 父区域拆出的水平邻字：跳过
            pa = ia.get("parent_id") if isinstance(ia, dict) else None
            pb = ib.get("parent_id") if isinstance(ib, dict) else None
            if pa and pb and pa == pb and ignore_horizontal_neighbors and _is_h_neighbor(ba, bb):
                continue
            # 同行左右相邻的正常多位数不记重叠；但 keep_pair 叠印对要记
            if ignore_horizontal_neighbors and _is_h_neighbor(ba, bb):
                ka = ia.get("keep_pair") if isinstance(ia, dict) else False
                kb = ib.get("keep_pair") if isinstance(ib, dict) else False
                if not (ka and kb):
                    continue

            inter = _bbox_intersection_area(ba, bb)
            iou = _bbox_iou(ba, bb)
            area_a = max(0.0, (ba[2] - ba[0]) * (ba[3] - ba[1]))
            area_b = max(0.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
            min_area = min(area_a, area_b)
            ratio = (inter / min_area) if min_area > 0 else 0.0
            overlapped = (
                inter >= min_intersection
                and iou >= iou_thr
                and ratio >= min_overlap_ratio
            )
            if overlapped:
                pairs += 1
                for b, rid in ((ba, ida), (bb, idb)):
                    key = tuple(b)
                    if key not in seen_box:
                        seen_box.add(key)
                        hit_boxes.append(b)
                        hit_ids.append(rid)
    return pairs, hit_boxes, hit_ids


def eval_logic(logic: dict[str, Any], facts: dict[str, Any]) -> tuple[bool, Any, Any]:
    """返回 (ok, actual, expected)。未知算子抛错。"""
    if not isinstance(logic, dict) or "op" not in logic:
        raise RuleEngineError(f"非法 logic: {logic}")
    op = logic["op"]

    if op == "and":
        actuals = []
        for arg in logic.get("args", []):
            ok, actual, expected = eval_logic(arg, facts)
            actuals.append(actual)
            if not ok:
                return False, actual, expected
        return True, actuals, "all true"

    if op == "or":
        last_actual, last_expected = None, None
        for arg in logic.get("args", []):
            ok, actual, expected = eval_logic(arg, facts)
            last_actual, last_expected = actual, expected
            if ok:
                return True, actual, expected
        return False, last_actual, last_expected

    if op == "not":
        ok, actual, expected = eval_logic(logic["arg"], facts)
        return (not ok), actual, f"not({expected})"

    path = logic.get("path")
    if not path:
        raise RuleEngineError(f"算子 {op} 缺少 path")
    actual = get_by_path(facts, path)

    if op == "not_empty":
        return (not _is_empty(actual)), actual, "非空"
    if op == "eq":
        expected = logic.get("value")
        return actual == expected, actual, expected
    if op == "neq":
        expected = logic.get("value")
        return actual != expected, actual, f"!={expected}"
    if op == "in":
        expected = logic.get("value")
        if not isinstance(expected, (list, tuple, set)):
            raise RuleEngineError("in.value 必须是列表")
        return actual in expected, actual, list(expected)
    if op == "regex":
        pattern = logic.get("pattern")
        if pattern is None:
            raise RuleEngineError("regex 缺少 pattern")
        if actual is None:
            return False, actual, pattern
        text = actual if isinstance(actual, str) else str(actual)
        return bool(re.search(pattern, text)), actual, pattern
    if op == "count_gte":
        count = logic["count"]
        n = len(actual) if isinstance(actual, (list, dict, str)) else 0
        return n >= count, n, f">={count}"
    if op == "count_lte":
        count = logic["count"]
        n = len(actual) if isinstance(actual, (list, dict, str)) else 0
        return n <= count, n, f"<={count}"
    if op == "bbox_overlap_count_lte":
        # path 指向带 bbox 的对象列表（如 annotations）
        items = actual if isinstance(actual, list) else ([] if actual is None else [actual])
        iou_thr = float(logic.get("iou_thr", 0.0))
        min_intersection = float(logic.get("min_intersection", 1.0))
        min_overlap_ratio = float(logic.get("min_overlap_ratio", 0.0))
        min_box_side = float(logic.get("min_box_side", 0.0))
        ignore_h = bool(logic.get("ignore_horizontal_neighbors", False))
        only_keep_pair = bool(logic.get("only_keep_pair", False))
        pairs, hit_boxes, hit_ids = _count_bbox_overlaps(
            items,
            iou_thr=iou_thr,
            min_intersection=min_intersection,
            min_overlap_ratio=min_overlap_ratio,
            min_box_side=min_box_side,
            ignore_horizontal_neighbors=ignore_h,
            only_keep_pair=only_keep_pair,
        )
        limit = int(logic["count"])
        ok = pairs <= limit
        return (
            ok,
            {
                "overlap_pairs": pairs,
                "evidence_bboxes": hit_boxes,
                "related_ids": hit_ids,
            },
            f"重叠对数<={limit}",
        )
    if op == "ref_exists":
        # path 指向引用 id；在 facts 常见集合中查找
        ref_id = actual
        if _is_empty(ref_id):
            return False, ref_id, "存在引用对象"
        pools = []
        for key in ("components", "features", "annotations", "datums"):
            pools.extend(facts.get(key) or [])
        tb = facts.get("title_block")
        if isinstance(tb, dict):
            pools.append(tb)
        for obj in pools:
            if not isinstance(obj, dict):
                continue
            if obj.get("id") == ref_id or obj.get("instance_id") == ref_id:
                return True, ref_id, "存在引用对象"
        return False, ref_id, "存在引用对象"

    raise RuleEngineError(f"未知算子: {op}")


def _evidence_bboxes(facts: dict[str, Any], path: str) -> list[list[float]]:
    root = path.split(".", 1)[0]
    node = facts.get(root)
    boxes: list[list[float]] = []
    if isinstance(node, dict) and node.get("bbox"):
        boxes.append(list(node["bbox"]))
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and item.get("bbox"):
                boxes.append(list(item["bbox"]))
    return boxes


def _related_ids(facts: dict[str, Any], path: str) -> list[str]:
    root = path.split(".", 1)[0]
    node = facts.get(root)
    ids: list[str] = []
    if isinstance(node, dict):
        rid = node.get("instance_id") or node.get("id")
        if rid:
            ids.append(str(rid))
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict):
                rid = item.get("instance_id") or item.get("id")
                if rid:
                    ids.append(str(rid))
    return ids


def evaluate(
    facts: dict[str, Any],
    rules: list[dict[str, Any]],
    *,
    rule_set_id: str = "default",
    perception_backend: str | None = None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for rule in rules:
        if rule.get("status") and rule.get("status") != "active":
            continue
        logic = rule.get("logic")
        if not logic:
            raise RuleEngineError(f"规则 {rule.get('rule_id')} 缺少 logic")
        ok, actual, expected = eval_logic(logic, facts)
        if ok:
            continue
        path = logic.get("path") or _first_path(logic) or ""
        if isinstance(actual, dict) and "evidence_bboxes" in actual:
            evidence = list(actual.get("evidence_bboxes") or [])
            related = list(actual.get("related_ids") or [])
            actual_out = actual.get("overlap_pairs", actual)
        else:
            evidence = _evidence_bboxes(facts, path) if path else []
            related = _related_ids(facts, path) if path else []
            actual_out = actual
        findings.append(
            {
                "rule_id": rule["rule_id"],
                "severity": rule.get("severity", "error"),
                "message": rule.get("message", rule.get("name", "")),
                "path": path,
                "actual": actual_out,
                "expected": expected,
                "evidence_bboxes": evidence,
                "related_ids": related,
            }
        )

    has_error = any(f["severity"] == "error" for f in findings)
    return {
        "passed": not has_error,
        "rule_set_id": rule_set_id,
        "rule_set_version": _max_version(rules),
        "perception_backend": perception_backend or facts.get("meta", {}).get("perception_backend"),
        "drawing_id": facts.get("drawing_id"),
        "findings": findings,
    }


def _first_path(logic: dict[str, Any]) -> str | None:
    if logic.get("path"):
        return logic["path"]
    if logic.get("op") == "not" and isinstance(logic.get("arg"), dict):
        return _first_path(logic["arg"])
    for arg in logic.get("args") or []:
        p = _first_path(arg)
        if p:
            return p
    return None


def _max_version(rules: list[dict[str, Any]]) -> int | None:
    versions = [r.get("version") for r in rules if isinstance(r.get("version"), int)]
    return max(versions) if versions else None
