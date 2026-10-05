"""Export registered research summaries only; never run experiments or change sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

MAX_SOURCE_BYTES = 262144
REPORT_NAME = re.compile(
    r"(?:REPORT|RESULT|FINDING|AUDIT|DECISION|DIAGNOSIS|RESEARCH_FACTS|"
    r"MECHANISM_SYNTHESIS|METRIC_VALUE_ASSESSMENT|PRIORITY_ASSESSMENT|"
    r"LOSS_OBJECTIVE_ANALYSIS|README|STATUS)", re.I)
EXCLUDE_NAME = re.compile(r"PROTOCOL|PLAN|USAGE|^upstream_|before_", re.I)
LINK = re.compile(r"!?\[([^\]\n]*)\]\(([^\n]*?)\)")
ACCESS_LINE = re.compile(
    r"(?i)(?:\bssh\s|\bscp\s|root@|connect\.[\w.-]+|"
    r"(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]|"
    r"(?:密码|口令)\s*[:：=])")
TOKEN = re.compile(r"(?:gh[pousr]_[\w]{20,}|github_pat_[\w]{20,}|"
                   r"sk-(?:proj-)?[\w-]{25,}|AKIA[A-Z0-9]{16})")
STANDALONE_SECRET = re.compile(r"^[ \t]*[A-Za-z0-9]{10,80}[ \t]*$", re.M)
ABS_WIN = re.compile(r"(?:/?[A-Za-z]:[/\\])[^\s<>\"|]*")
ABS_REMOTE = re.compile(r"/(?:root|home|mnt|tmp|autodl-tmp)/[^\s<>\"|]*")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(text: str, root: Path) -> str:
    # Only publish research prose, never operational access details.
    text = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
                  "[已移除访问凭据]", text, flags=re.S)
    text = "\n".join("[已移除连接或访问凭据行]" if ACCESS_LINE.search(line) else line
                     for line in text.splitlines())
    text = TOKEN.sub("[已移除访问凭据]", text)
    text = STANDALONE_SECRET.sub("[已移除独立标识或凭据行]", text)
    for prefix in (str(root), root.as_posix(), "C:/Dpan/codexproject/paper-disc",
                   "C:\\Dpan\\codexproject\\paper-disc"):
        text = text.replace(prefix + "/", "").replace(prefix + "\\", "")
    text = ABS_WIN.sub("[机器路径已省略]", text)
    text = ABS_REMOTE.sub("[远端路径已省略]", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[联系信息已省略]", text)
    return text


def clean_value(value, root: Path):
    if isinstance(value, str):
        return sanitize(value, root)
    if isinstance(value, list):
        return [clean_value(v, root) for v in value]
    if isinstance(value, dict):
        return {k: clean_value(v, root) for k, v in value.items()
                if not re.search(r"(?i)password|passwd|secret|token|command|ssh|credential", k)}
    return value


def quote_excerpt(body: str) -> str:
    # Literal excerpt, never turn a historical author's wording into a new claim.
    lines = body.splitlines()
    start = next((i for i, x in enumerate(lines)
                  if x.startswith("#") and re.search("结论先行|结论与|最终结论|核心结论|结论|结论摘要", x)), 0)
    chosen = []
    for line in lines[start:]:
        if chosen and line.startswith("## ") and len(chosen) > 2:
            break
        chosen.append(line)
        if sum(len(x) for x in chosen) > 2200:
            break
    excerpt = "\n".join(chosen).strip()
    excerpt = LINK.sub(lambda m: m.group(1), excerpt)
    return "\n".join("> " + line for line in excerpt.splitlines())


def linkify(body: str, source: Path, mapping: dict, root: Path) -> str:
    def replace(match):
        label, target = match.groups()
        target = target.strip().strip("<>")
        if re.match(r"https?://", target):
            return match.group(0)
        if target.startswith("#"):
            return match.group(0)
        raw, _, anchor = target.partition("#")
        raw = re.sub(r":\d+$", "", unquote(raw))
        if re.match(r"/?[A-Za-z]:[/\\]", raw):
            candidate = Path(raw.lstrip("/")).resolve()
        elif raw.startswith(("experiments/", "project/", "paper/", "shared/")):
            candidate = (root / raw).resolve()
        else:
            candidate = (source.parent / raw).resolve()
        dest = mapping.get(str(candidate))
        if dest:
            # All copied reports live at reports/<study>/<file>.
            target_link = "../../" + dest + ("#" + anchor if anchor else "")
            return f"[{label}]({quote(target_link, safe='/#')})"
        return f"{label}（本地证据；本概述仓库未收录）"
    return LINK.sub(replace, body)


def collect_runs(directory: Path, root: Path):
    result = []
    runs = directory / "runs"
    if not runs.exists():
        return result
    for path in sorted(runs.glob("*/run.json")):
        try:
            obj = read_json(path)
        except (ValueError, OSError):
            result.append({"directory": path.parent.name, "status": "unreadable_record"})
            continue
        fields = ("run_id", "id", "status", "execution_status", "artifact_status",
                  "transfer_status", "started_at", "finished_at", "ended_at", "exit_code", "seed")
        row = {key: clean_value(obj[key], root) for key in fields if key in obj}
        row["directory"] = path.parent.name
        result.append(row)
    return result


def export(root: Path, destination: Path, context_path: Path):
    root, destination = root.resolve(), destination.resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("Export must be a separate sibling checkout, outside the research source tree.")
    if not (root / "experiments").is_dir():
        raise ValueError("Research root not found")
    marker = destination / "PUBLICATION.json"
    if destination.exists() and any(p.name != ".git" for p in destination.iterdir()) and not marker.exists():
        raise ValueError("Nonempty export destination has no publication marker")
    previous = read_json(marker).get("generated_files", []) if marker.exists() else []
    context = read_json(context_path)
    generated, sources, skipped, redactions = {}, [], [], []
    studies, reports, mapping = [], [], {}
    identities = set()
    for directory in sorted((root / "experiments").iterdir()):
        meta_path = directory / "study.json"
        if not directory.is_dir() or not meta_path.is_file():
            continue
        meta = read_json(meta_path)
        identity = meta.get("study_id") or meta.get("id")
        if not identity or identity in identities:
            raise ValueError(f"Missing or duplicate direct Study identity: {directory.name}")
        identities.add(identity)
        reviewed = context.get("studies", {}).get(directory.name, {})
        selected, selected_paths, selected_targets = [], set(), {}

        def add_report(path: Path):
            # Explicit references may reach Run reports, but never another
            # study, an external symlink, or arbitrary machine files.
            path = path.resolve()
            if directory.resolve() not in path.parents:
                raise ValueError(f"Report is outside its Study: {directory.name}: {path}")
            if not path.is_file():
                raise ValueError(f"Explicit report does not exist: {directory.name}: {path}")
            if path.suffix.lower() not in (".md", ".json"):
                raise ValueError(f"Only Markdown/JSON reports may be exported: {path}")
            if path in selected_paths:
                return
            selected_paths.add(path)
            if path.stat().st_size > MAX_SOURCE_BYTES:
                skipped.append({"source": path.relative_to(root).as_posix(), "reason": "source_size_limit"})
                return
            # linkify assumes every report is exactly reports/<study>/<file>.
            relative = path.relative_to(directory.resolve())
            filename = "__".join(relative.parts)
            target = f"reports/{directory.name}/{filename}"
            if target in selected_targets and selected_targets[target] != path:
                raise ValueError(f"Flattened report path collision: {target}")
            selected_targets[target] = path
            mapping[str(path)] = target
            selected.append({"path": path, "target": target, "is_json": path.suffix.lower() == ".json"})
            reports.append(selected[-1])

        def add_explicit(reference, origin):
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"{directory.name}: {origin} must be a nonempty relative report path")
            relative = Path(reference.replace("\\", "/"))
            if relative.is_absolute() or relative.drive or relative.root:
                raise ValueError(f"{directory.name}: {origin} must be relative to its Study")
            add_report(directory / relative)

        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            is_md = path.suffix.lower() == ".md" and REPORT_NAME.search(path.name) and not EXCLUDE_NAME.search(path.name)
            is_json = path.name.lower() == "summary.json"
            if not (is_md or is_json):
                continue
            add_report(path)
        for key in ("report", "interpretation"):
            if meta.get(key) is not None:
                add_explicit(meta[key], f"study.json {key}")
        included = reviewed.get("include_reports", [])
        if not isinstance(included, list):
            raise ValueError(f"{directory.name}: include_reports must be a list of relative report paths")
        for reference in included:
            add_explicit(reference, "context include_reports")
        studies.append({"directory": directory.name, "study_id": identity,
                        "title": reviewed.get("title") or meta.get("title") or meta.get("name") or directory.name,
                        "kind": meta.get("kind") or "unknown", "status": meta.get("status") or "unknown",
                        "description": meta.get("description") or meta.get("provenance") or "",
                        "scope": meta.get("scope") or "未在登记中声明",
                        "reports": selected, "runs": collect_runs(directory, root), "reviewed": reviewed})
        sources.append({"source": meta_path.relative_to(root).as_posix(), "sha256": sha(meta_path)})
    for report in reports:
        path = report["path"]
        if report["is_json"]:
            try:
                parsed = read_json(path)
            except (ValueError, OSError):
                skipped.append({"source": path.relative_to(root).as_posix(), "reason": "unreadable_summary"})
                mapping.pop(str(path.resolve()), None)
                continue
            body = json.dumps(clean_value(parsed, root), ensure_ascii=False, indent=2) + "\n"
        else:
            original = path.read_text(encoding="utf-8-sig")
            body = sanitize(linkify(original, path, mapping, root), root)
            header = (f"> 导出的源报告快照，保持原适用范围；不是新的研究结论。\n"
                      f"> 来源：`{path.relative_to(root).as_posix()}`；SHA256：`{sha(path)}`。\n"
                      f"> 访问凭据、机器路径及未上传资产链接已省略。\n\n")
            body = header + body + "\n"
            if "已移除" in body:
                redactions.append(path.relative_to(root).as_posix())
        generated[report["target"]] = body
        report["body"] = body
        sources.append({"source": path.relative_to(root).as_posix(), "sha256": sha(path),
                        "export": report["target"]})
    index = ["# 实验索引", "", f"共 {len(studies)} 个直接登记的研究单元。数量不是完成实验或独立结论的数量。",
             "只扫描 `experiments/*/study.json`；备份、快照和 Run 不重复计作新实验。",
             "历史导入、失败尝试、冒烟测试和缺少报告的单元都保留，不按目录名补造结论。", "",
             "开始阅读前先看 [当前研究状态](CURRENT_STATUS.md)。", "",
             "| 研究单元 | 类型 | 登记状态 | 收录报告/统计 | 已登记 Run |", "|---|---|---|---:|---:|"]
    catalog = []
    for study in studies:
        slug, identity, reviewed = study["directory"], study["study_id"], study["reviewed"]
        present = [r for r in study["reports"] if r["target"] in generated]
        destination_md = f"experiments/{slug}.md"
        title = sanitize(str(study["title"]), root)
        body = [f"# {title}", "", f"- Study ID：`{identity}`", f"- 来源目录：`experiments/{slug}/`",
                f"- 类型：`{study['kind']}`；登记状态：`{study['status']}`。",
                f"- 原登记范围：{sanitize(str(study['scope']), root)}。",
                f"- 收录 {len(present)} 份报告/统计快照；发现 {len(study['runs'])} 个已登记 Run。",
                "", "登记状态仅来自元数据；`imported` 表示历史导入，`unknown` 表示未记录，不能据此认定成功或失败。", ""]
        if study["description"]:
            body += ["原登记说明：" + sanitize(str(study["description"]), root), ""]
        if reviewed.get("summary"):
            body += ["## 经过范围核对的概述", "", reviewed["summary"], ""]
        else:
            body += ["## 概述与证据状态", "", "本单元未在本次发布中重新进行科学审查。下列记录仅重用现有报告或统计，不把历史结果自动升级为当前结论。", ""]
        md_reports = [r for r in present if not r["is_json"]]
        if md_reports and not reviewed.get("summary"):
            primary = min(md_reports, key=lambda r: (0 if r["path"].name == "REPORT.md" else 1 if r["path"].name in ("OFFICIAL_RESULTS.md", "RESULTS.md") else 2, r["path"].name))
            # Strip the generated provenance prefix before quoting the source.
            report_body = primary["body"].split("\n\n", 1)[-1]
            body += [f"源报告 `{primary['path'].name}` 的原文节选（不替代全文及其限制）：", "", quote_excerpt(report_body), ""]
        elif not present:
            body += ["未找到符合发布范围的报告或 `SUMMARY.json`。可能仅有训练产物、历史记录或尚未回传结果；当前不能据此提供经核实的实验结论。", ""]
        elif not md_reports:
            body += ["仅收录现有机器统计。它记录数值或执行状态，不等于已解释机制；样本、协议与结论应以原实验补充材料核对。", ""]
        body += ["## 可读取的源证据", ""]
        for report in present:
            body.append(f"- [{report['path'].name}](../{quote(report['target'], safe='/')})")
        if not present:
            body.append("- 无已收录报告；保留该 Study 以避免误认为尚未开展而重复实验。")
        if study["runs"]:
            body += ["", "## 已登记的运行", "", "此表只摘录 Run 元数据，不导出命令、训练日志、账号或远端地址。未列出的执行/产物/回传状态保持未知。", "",
                     "```json", json.dumps(study["runs"], ensure_ascii=False, indent=2), "```"]
        body += ["", "[返回实验索引](../EXPERIMENT_INDEX.md)", ""]
        generated[destination_md] = "\n".join(body)
        index.append(f"| [{title.replace('|', '／')}]({quote(destination_md, safe='/')}) | {study['kind']} | {study['status']} | {len(present)} | {len(study['runs'])} |")
        catalog.append({k: study[k] for k in ("directory", "study_id", "title", "kind", "status", "scope", "runs")} |
                       {"summary": destination_md, "reports": [r["target"] for r in present]})
    stats = {"registered_studies": len(studies), "studies_with_reports": sum(bool(s["reports"]) for s in studies),
             "exported_source_reports": len([p for p in generated if p.startswith("reports/")]),
             "registered_runs": sum(len(s["runs"]) for s in studies),
             "kinds": dict(Counter(s["kind"] for s in studies)),
             "statuses": dict(Counter(s["status"] for s in studies))}
    generated["EXPERIMENT_INDEX.md"] = "\n".join(index) + "\n"
    generated["README.md"] = (context_path.parent / context["readme_file"]).read_text(encoding="utf-8-sig").replace("{{STUDIES}}", str(len(studies)))
    generated["CURRENT_STATUS.md"] = (context_path.parent / context["current_status_file"]).read_text(encoding="utf-8-sig")
    generated["catalog.json"] = json.dumps({"schema_version": 1, "statistics": stats, "studies": catalog}, ensure_ascii=False, indent=2) + "\n"
    generated["SOURCE_MANIFEST.json"] = json.dumps({"scope": "direct registered studies; no backup duplication", "sources": sources, "skipped": skipped, "redacted_reports": redactions}, ensure_ascii=False, indent=2) + "\n"
    generated[".gitignore"] = "# Only the generated documentation snapshot belongs in this repository.\n*.pt\n*.pth\n*.npy\n*.npz\n*.pkl\n*.zip\n*.jpg\n*.png\n__pycache__/\n.env*\n"
    generated["PUBLICATION.json"] = json.dumps({"schema_version": 1, "generated_at": datetime.now().astimezone().isoformat(), "repository": context["repository"], "statistics": stats, "generated_files": sorted(list(generated) + ["PUBLICATION.json"])}, ensure_ascii=False, indent=2) + "\n"
    # Validate links and secrets before writing, committing, or pushing anything.
    for name, body in generated.items():
        if TOKEN.search(body) or re.search(r"root@|connect\.[\w.-]+|PRIVATE KEY|\bssh\s+-p\b", body, re.I):
            raise ValueError(f"Credential/access material in generated {name}")
        if name.endswith(".md"):
            for m in LINK.finditer(body):
                url = unquote(m.group(2).strip().strip("<>"))
                if url.startswith(("http://", "https://", "#")):
                    continue
                target = (Path(name).parent / url.split("#")[0]).as_posix()
                # Resolve dot segments without consulting disk (not written yet).
                target = (destination / target).resolve().relative_to(destination).as_posix()
                if target not in generated:
                    raise ValueError(f"Broken publication link: {name} -> {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in generated.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="\n")
    # Delete only a previous explicitly generated file, with a checked absolute target.
    for name in previous:
        if name in generated:
            continue
        path = (destination / name).resolve()
        if destination not in path.parents or ".git" in path.relative_to(destination).parts:
            raise ValueError("Unsafe old generated file path")
        if path.is_file():
            path.unlink()
    print(json.dumps({"statistics": stats, "files": len(generated), "bytes": sum(len(v.encode('utf-8')) for v in generated.values()), "skipped": len(skipped)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=Path, default=Path(__file__).with_name("publication_context.json"))
    args = parser.parse_args()
    export(args.project_root, args.output, args.context)
