import re
import os
import shutil
import json
import time
import logging
try:
    from xiyou.config import load_settings, resolve_paths, list_target_books
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from xiyou.config import load_settings, resolve_paths, list_target_books

settings = load_settings()
logging.basicConfig(level=logging.INFO, format="%(message)s")
_corpora = settings.get("corpora", {})
BOOK_KEY = _corpora.get("default") or (next(iter(_corpora.get("items", {}).keys())) if isinstance(_corpora.get("items", {}), dict) and _corpora.get("items", {}) else "xiyouji")
_paths_single = resolve_paths(settings, BOOK_KEY)
RAW_FILE_PATH = _paths_single["raw_file"]
OUTPUT_DIR = _paths_single["chapters_dir"]
FILENAME_MODE = settings["naming"]["mode"]

# 正则表达式：匹配章节标题 (与V2版相同，已优化)
CHAPTER_PATTERN = re.compile(r"^\s*第[零一二三四五六七八九十百千]+回.*")
# 结束标记：看到这一行就停止处理（Gutenberg 的标准结尾）
END_MARKER = "*** END OF THE PROJECT GUTENBERG"

# 非法文件名字符
ILLEGAL_CHARS = r'[\\/:*?"<>|]'

class ChapterSplitter:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.chapters = []
        self.current_buffer = []
        self.current_title = None

    def _sanitize_filename(self, title):
        """净化标题，去除操作系统不允许的字符"""
        # 1. 替换非法字符为空格/下划线
        sanitized = re.sub(ILLEGAL_CHARS, '_', title)
        # 2. 去除多余的空格
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        # 3. 截断防止文件名过长 (例如限制到 100 个字符)
        return sanitized[:100]

    def _get_filename(self, chapter_num, title):
        """根据配置模式生成文件名"""
        num_prefix = f"{chapter_num:03d}"

        if FILENAME_MODE == "NUMBERING":
            # 模式 1: 纯数字 (e.g., chapter_001.txt)
            return f"chapter_{num_prefix}.txt"

        elif FILENAME_MODE == "TITLE_PREFIXED":
            # 模式 2: 序号前缀 + 净化后的标题 (e.g., 001_第一回 灵根育孕源流出.txt)
            sanitized_title = self._sanitize_filename(title)
            # 确保标题不为空，否则使用默认值
            title_part = sanitized_title if sanitized_title else "Chapter"
            return f"{num_prefix}_{title_part}.txt"

        else:
            # 默认 fallback
            return f"unknown_{num_prefix}.txt"

    def _save_chapter(self):
        if not self.current_title or not self.current_buffer:
            return

        chapter_num = len(self.chapters) + 1

        # 1. 生成文件名
        filename = self._get_filename(chapter_num, self.current_title)
        file_path = os.path.join(self.output_dir, filename)

        content = "\n".join(self.current_buffer).strip()

        if len(content) > 50:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.chapters.append(self.current_title)
            logging.info(json.dumps({"stage": "split", "event": "save", "file": filename, "length": len(content)}, ensure_ascii=False))

        self.current_buffer = []

    def run(self):
        start = time.time()
        # 1. 准备目录 (如果目录不存在，则创建)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 2. 【核心优化：安全清理旧文件】
        logging.info(json.dumps({"stage": "split", "event": "cleanup_start", "dir": self.output_dir}, ensure_ascii=False))

        # 正则匹配两种命名模式的文件：chapter_XXX.txt 或 XXX_标题.txt
        # \d{3}_ 匹配 001_
        cleanup_pattern = re.compile(r"^(chapter_\d{3}|\d{3}_.*)\.txt$")

        cleaned_count = 0
        for filename in os.listdir(self.output_dir):
            if cleanup_pattern.match(filename):
                file_to_delete = os.path.join(self.output_dir, filename)
                os.remove(file_to_delete)
                cleaned_count += 1

        logging.info(json.dumps({"stage": "split", "event": "cleanup_done", "deleted": cleaned_count}, ensure_ascii=False))
        logging.info(json.dumps({"stage": "split", "event": "start", "input": self.input_path, "mode": FILENAME_MODE}, ensure_ascii=False))

        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                started = False

                for line in f:
                    line = line.rstrip()

                    if END_MARKER in line:
                        break

                    if CHAPTER_PATTERN.match(line) and len(line) < 50:
                        if started:
                            self._save_chapter()

                        started = True
                        self.current_title = line.strip()
                        self.current_buffer = [self.current_title]
                        continue

                    if started:
                        self.current_buffer.append(line)

                if started:
                    self._save_chapter()

        except FileNotFoundError:
            logging.error(json.dumps({"stage": "split", "event": "error", "error_type": "FileNotFound", "input": self.input_path}, ensure_ascii=False))
            return
        duration = int((time.time() - start) * 1000)
        logging.info(json.dumps({"stage": "split", "event": "done", "chapters": len(self.chapters), "duration_ms": duration}, ensure_ascii=False))


# ================= 执行入口 =================
if __name__ == "__main__":
    mode = settings.get("corpora", {}).get("mode", "single")
    if mode == "batch":
        books = list_target_books(settings)
        for bk in books:
            p = resolve_paths(settings, bk)
            logging.info(json.dumps({"stage": "split", "event": "book_start", "book_key": bk, "input": p.get("raw_file"), "output_dir": p.get("chapters_dir")}, ensure_ascii=False))
            splitter = ChapterSplitter(p["raw_file"], p["chapters_dir"])
            splitter.run()
    else:
        splitter = ChapterSplitter(RAW_FILE_PATH, OUTPUT_DIR)
        splitter.run()
