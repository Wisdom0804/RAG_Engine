# -*- coding: utf-8 -*-
"""
文件名：splitter_base.py
文件描述: 文档分块基类，使用策略模式提供多种切分方法，目前只切文本
作者: 郑智文
创建日期: 2026/9/5 12:11
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
import re
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer

from app.libs.enums import ChunkingStrategyEnum


class ChunkingStrategy(ABC):
    """分块策略抽象基类，所有具体切分策略均需实现 split 与 configure 方法"""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """
        切分文本，返回分块列表
        :param text: 待切分的文本
        :return: 分块列表
        """
        pass

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """
        批量设置 __init__ 中的构造参数，未传入的参数保持原值不变
        :param kwargs: 与各策略 __init__ 同名的可选参数
        """
        pass


class FixedLengthChunking(ChunkingStrategy):
    """
    定长切分策略
    按固定长度对文本进行切分，不考虑语义完整性
    overlap 为相邻分块共享的绝对字符数（overlap=10 表示相邻分块重叠 10 个字符），
    overlap 取自前一分块尾部，从 chunk_size 内部扣除，保证每个分块长度不超过 chunk_size
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 0):
        """
        :param chunk_size: 每个分块的最大字符长度
        :param overlap: 相邻分块间的重叠字符长度，默认 0 表示不重叠
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._validate()

    def _validate(self) -> None:
        """原子校验 chunk_size 与 overlap 的跨字段不变量"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_size <= self.overlap:
            raise ValueError(f"overlap 必须在[0,{self.chunk_size})之间")

    def configure(self, *, chunk_size: int = None, overlap: int = None) -> None:
        """
        批量设置构造参数，未传入的不修改
        :param chunk_size: 每个分块的最大字符长度
        :param overlap: 重叠字符长度
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if overlap is not None:
            self.overlap = overlap
        self._validate()

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        # 步长 = chunk_size - overlap，保证相邻分块共享 overlap 个字符且每片不超限
        step = self.chunk_size - self.overlap
        while start < text_len:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= text_len:
                break
            start += step
        return chunks


class SemanticChunking(ChunkingStrategy):
    """
    语义切分策略
    不固定分块大小，通过嵌入函数计算相邻句子的相似度，相似度低于阈值时断开形成新分块
    默认嵌入模型需要科学上网，自定义嵌入方法需要重写encode方法
    通过 max_chunk_size 限制最大分片大小，防止语义高度连贯时产生超大分片
    """

    def __init__(self, embed_fn=None, threshold: float = 0.5, max_chunk_size: int = 1000):
        """
        :param embed_fn: 嵌入函数/模型，需支持 encode(list[str]) -> ndarray，
                         默认使用 SentenceTransformer("all-MiniLM-L6-v2")
        :param threshold: 相似度阈值，低于该值则断开，取值范围 [0, 1]
        :param max_chunk_size: 单个分块的最大字符长度，超过则按句子进一步切分（句子仍超长则定长兜底）
        """
        if embed_fn is None:
            embed_fn = SentenceTransformer("all-MiniLM-L6-v2")
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size
        self._validate()

    def configure(self, *, embed_fn=None, threshold: float = None, max_chunk_size: int = None) -> None:
        """
        批量设置构造参数，未传入的不修改
        :param embed_fn: 嵌入函数/模型，需支持 encode(list[str]) -> ndarray
        :param threshold: 相似度阈值，低于该值则断开，取值范围 [0, 1]
        :param max_chunk_size: 单个分块的最大字符长度
        """
        if embed_fn is not None:
            self.embed_fn = embed_fn
        if threshold is not None:
            self.threshold = threshold
        if max_chunk_size is not None:
            self.max_chunk_size = max_chunk_size
        self._validate()

    def _validate(self) -> None:
        """原子校验 threshold 与 max_chunk_size 的取值范围"""
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("threshold 必须在 [0, 1] 范围内")
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size 必须大于 0")

    def _split_sentences(self, text: str) -> list[str]:
        """将文本切分为句子列表"""
        sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
        return [s for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _split_oversized(self, text: str) -> list[str]:
        """
        对超大分片按句子切分后逐句拼接以逼近 max_chunk_size，
        单句仍超长时用定长切分兜底
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        chunks = []
        current = ''
        for sent in sentences:
            if len(sent) > self.max_chunk_size:
                # 先收尾当前累积内容
                if current:
                    chunks.append(current)
                    current = ''
                # 单句超长，定长切分兜底
                for i in range(0, len(sent), self.max_chunk_size):
                    chunks.append(sent[i:i + self.max_chunk_size])
            elif len(current) + len(sent) <= self.max_chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        # 切分
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # 批量计算所有句子的嵌入向量
        embeddings = self.embed_fn.encode(sentences)

        chunks = []
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                # 相似度低于阈值，断开形成新分块
                chunks.append(''.join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            chunks.append(''.join(current))

        # 限制最大分片大小，防止语义连贯产生超大分片
        result = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                result.extend(self._split_oversized(chunk))
            else:
                result.append(chunk)
        return result


class RecursiveChunking(ChunkingStrategy):
    """
    递归切分策略
    依次尝试更细粒度的分隔符对文本进行切分，直到每个分片不超过 chunk_size。
    分隔符粒度由粗到细：段落 → 句子 → 标点 → 空格 → 定长。
    最后将过小的分片拼接以逼近目标 chunk_size，相邻分片保留一定 overlap。
    """

    # 默认分隔符序列，粒度由粗到细；None 表示定长兜底
    DEFAULT_SEPARATORS = [
        '\n\n',                       # 段落（空行分隔）
        '\n',                         # 换行
        r'(?<=[。！？.!?])',           # 句子
        r'(?<=[,，;；:：、])',         # 标点
        r'\s',                        # 空格
        None,                         # 定长兜底
    ]

    def __init__(self, chunk_size: int = 500, overlap: int = 50, separators: list = None):
        """
        :param chunk_size: 每个分块的目标/最大字符长度
        :param overlap: 相邻分块间的重叠字符长度
        :param separators: 自定义分隔符序列，粒度由粗到细，最后一项为 None 时表示定长兜底
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else list(self.DEFAULT_SEPARATORS)
        self._validate()

    def _validate(self) -> None:
        """原子校验 chunk_size 与 overlap 的取值范围"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError(f"overlap 必须在 [0, {self.chunk_size}) 之间")

    def configure(self, chunk_size: int = None, overlap: int = None, separators: list = None) -> None:
        """
        批量设置构造参数，未传入的不修改
        :param chunk_size: 每个分块的目标/最大字符长度
        :param overlap: 相邻分块间的重叠字符长度
        :param separators: 自定义分隔符序列
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if overlap is not None:
            self.overlap = overlap
        if separators is not None:
            self.separators = separators
        self._validate()

    def _split_by_separator(self, text: str, separator) -> list[str]:
        """按单个分隔符切分文本，保留分隔产生的非空片段"""
        if separator is None:
            # 定长兜底
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        pieces = re.split(separator, text)
        return [p for p in pieces if p]

    def _recursive_split(self, text: str, sep_idx: int = 0) -> list[str]:
        """
        递归切分：用更细的分隔符处理超长片段，返回小分片列表。
        若当前分隔符切不动（仍只有一个片段且超长），则换更细的分隔符继续。
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []
        if sep_idx >= len(self.separators):
            # 全部分隔符都已尝试，定长兜底
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = self.separators[sep_idx]
        pieces = self._split_by_separator(text, separator)
        # 当前分隔符切不动，换更细的分隔符
        if len(pieces) <= 1:
            return self._recursive_split(text, sep_idx + 1)

        result = []
        for piece in pieces:
            if len(piece) > self.chunk_size:
                # 该片段仍超长，递归用更细的分隔符
                result.extend(self._recursive_split(piece, sep_idx + 1))
            else:
                result.append(piece)
        return result

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """将小分片拼接以逼近 chunk_size，相邻分块以 overlap 衔接"""
        if not chunks:
            return []
        merged = []
        current = ''
        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                # 当前累积已达上限，收尾并开启新分块
                merged.append(current)
                if self.overlap > 0:
                    tail = current[-self.overlap:]
                    # overlap + chunk 仍在上限内则带上 overlap
                    if len(tail) + len(chunk) <= self.chunk_size:
                        current = tail + chunk
                    else:
                        current = chunk
                else:
                    current = chunk
        if current:
            merged.append(current)
        return merged

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        small_chunks = self._recursive_split(text)
        return self._merge_chunks(small_chunks)


class StructuralChunking(ChunkingStrategy):
    """
    结构切分策略
    接收文档解析后的文本，按文档类型（markdown / html / docx / code）识别结构单元，
    在结构内部递归切分，并在 chunk 头部保留标题路径等元数据，相邻分块保留 overlap。
    单个结构单元超长时，借助 RecursiveChunking 在结构内进一步切分。
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, doc_type: str = 'markdown'):
        """
        :param chunk_size: 每个分块的目标/最大字符长度
        :param overlap: 相邻分块间的重叠字符长度
        :param doc_type: 文档类型，可选 'markdown' / 'html' / 'docx' / 'code'
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.doc_type = doc_type
        self._recursive = RecursiveChunking(chunk_size=chunk_size, overlap=overlap)
        self._validate()

    def _validate(self) -> None:
        """原子校验 chunk_size、overlap 与 doc_type"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError(f"overlap 必须在 [0, {self.chunk_size}) 之间")
        if self.doc_type not in ('markdown', 'html', 'docx', 'code'):
            raise ValueError(f"不支持的文档类型 '{self.doc_type}'，可选: markdown/html/docx/code")

    def configure(self, *, chunk_size: int = None, overlap: int = None, doc_type: str = None) -> None:
        """
        批量设置构造参数，未传入的不修改
        :param chunk_size: 每个分块的目标/最大字符长度
        :param overlap: 相邻分块间的重叠字符长度
        :param doc_type: 文档类型
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if overlap is not None:
            self.overlap = overlap
        if doc_type is not None:
            self.doc_type = doc_type
        # 同步内部递归切分器参数
        self._recursive.configure(chunk_size=self.chunk_size, overlap=self.overlap)
        self._validate()

    # ---------- 各类型结构单元识别 ----------

    def _parse_markdown(self, text: str) -> list[dict]:
        """
        识别 markdown 结构单元：标题层级 + 标题下内容块。
        标题行仅用于更新标题路径 path，本身不作为内容单元（路径前缀已携带层级信息）；
        非标题行按段落归入当前标题下，作为内容单元输出。
        """
        units = []
        title_stack = []  # [(level, title), ...]
        content_lines = []

        def flush_content():
            if content_lines:
                path = ' > '.join(t for _, t in title_stack)
                units.append({'path': path, 'content': '\n'.join(content_lines)})
                content_lines.clear()

        for line in text.split('\n'):
            m = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
            if m:
                flush_content()
                level = len(m.group(1))
                title = m.group(2)
                # 弹出同级及更高级标题，维持层级栈
                while title_stack and title_stack[-1][0] >= level:
                    title_stack.pop()
                title_stack.append((level, title))
            else:
                content_lines.append(line)
        flush_content()
        return units

    def _parse_html(self, text: str) -> list[dict]:
        """
        识别 html 结构单元：h1~h6 标题与 p/div/section/li 等正文块。
        按文档顺序遍历标题与正文块，正文归入其前序最近标题路径下。
        """
        units = []
        title_stack = []  # [(level, title), ...]

        # 单次扫描：同时匹配标题与正文块，按出现顺序处理，保证正文归入前序标题
        token_re = re.compile(
            r'<(h[1-6])[^>]*>(.*?)</\1>|<(p|div|section|li)[^>]*>(.*?)</\3>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for m in token_re.finditer(text):
            if m.group(1):  # 标题：仅更新标题路径，不作为内容单元
                level = int(m.group(1)[-1])
                title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                while title_stack and title_stack[-1][0] >= level:
                    title_stack.pop()
                title_stack.append((level, title))
            else:  # 正文块
                inner = re.sub(r'<[^>]+>', '', m.group(4)).strip()
                if inner:
                    path = ' > '.join(t for _, t in title_stack)
                    units.append({'path': path, 'content': inner})
        return units

    def _parse_docx(self, text: str) -> list[dict]:
        """
        识别 docx 解析后的结构单元。
        约定解析输出中以形如「# 标题」「Heading 1: 标题」的行作为标题，
        其余行按段落归入当前标题下。
        """
        units = []
        title_stack = []
        content_lines = []

        def flush_content():
            if content_lines:
                path = ' > '.join(title_stack)
                units.append({'path': path, 'content': '\n'.join(content_lines).rstrip('\n')})
                content_lines.clear()

        for line in text.split('\n'):
            # 兼容 markdown 风格「# 标题」与「Heading N: 标题」风格
            md_m = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
            hd_m = re.match(r'^Heading\s*([1-6])\s*[:：]\s*(.+?)\s*$', line, flags=re.IGNORECASE)
            if md_m:
                level, title = len(md_m.group(1)), md_m.group(2)
            elif hd_m:
                level, title = int(hd_m.group(1)), hd_m.group(2)
            else:
                level, title = None, None

            if level is not None:
                flush_content()
                # 弹出同级及更深层标题，维持层级栈深度
                while title_stack and len(title_stack) >= level:
                    title_stack.pop()
                title_stack.append(title)
            else:
                content_lines.append(line)
        flush_content()
        return units

    def _parse_code(self, text: str) -> list[dict]:
        """
        识别代码结构单元：函数/类/方法定义作为分界，块内为正文。
        以缩进推断层级：同级缩进的定义为兄弟（平铺），更深缩进为子定义。
        兼容 Python(def/class)、JS/TS(function/class/export) 等常见定义。
        def/class 行与其后续函数体行合并为同一结构单元，避免签名与实现被拆开。
        """
        units = []
        title_stack = []  # [(indent, name), ...]
        current_lines = []

        def flush_content():
            if current_lines:
                path = ' > '.join(n for _, n in title_stack)
                units.append({'path': path, 'content': '\n'.join(current_lines)})
                current_lines.clear()

        # 函数/类/方法定义行：捕获前导空白（缩进）与定义名
        define_re = re.compile(
            r'^(\s*)(?:export\s+)?(?:async\s+)?(?:def|function|class)\s+(\w+)',
            re.MULTILINE,
        )
        for line in text.split('\n'):
            m = define_re.match(line)
            if m:
                flush_content()
                indent = len(m.group(1))
                name = m.group(2)
                # 弹出同级及更深缩进的定义，维持层级栈
                while title_stack and title_stack[-1][0] >= indent:
                    title_stack.pop()
                title_stack.append((indent, name))
                # def/class 行作为新单元的第一行，函数体后续累积到同一单元
                current_lines.append(line)
            else:
                current_lines.append(line)
        flush_content()
        return units

    # ---------- chunk 组装 ----------

    def _build_chunks(self, units: list[dict]) -> list[str]:
        """
        在结构内部递归切分并组装 chunk：
        - 在 chunk 头部保留标题路径元数据 [path]
        - 单个结构单元超长时用 RecursiveChunking 在结构内继续切分
        - 相邻分块保留 overlap（取自前一 chunk 的正文尾部）
        """
        chunks = []
        current = ''
        current_path = None

        for unit in units:
            path = unit['path']
            content = unit['content']
            if not content:
                continue

            # 标题路径变化时，若当前分块非空则收尾
            if path and path != current_path and current.strip():
                chunks.append(current)
                current = ''

            prefix = f"[{path}]\n" if path else ''
            piece = prefix + content

            if len(piece) <= self.chunk_size:
                if len(current) + len(piece) + (1 if current else 0) <= self.chunk_size:
                    current = current + '\n' + piece if current else piece
                else:
                    if current.strip():
                        chunks.append(current)
                    # overlap 衔接（代码切分禁用 char-level overlap）
                    # overlap 取前一片尾部，置于 prefix 之后、content 之前，保证路径前缀始终在 chunk 头部
                    if self.overlap > 0 and self.doc_type != 'code' and chunks:
                        tail = chunks[-1][-self.overlap:]
                        overlap_piece = prefix + tail + content
                        if len(overlap_piece) <= self.chunk_size:
                            current = overlap_piece
                        else:
                            current = piece
                    else:
                        current = piece
            else:
                # 单个结构单元超长，在结构内递归切分
                # 递归切分目标需扣除 prefix 长度，避免拼接后超限
                # 代码切分例外：目标不减 prefix，保证代码行完整，避免在子句边界拆碎语句
                # 递归切分内部关闭 overlap，避免字符级片段破坏代码/句子结构；chunk 间 overlap 由 _build_chunks 统一处理
                if self.doc_type == 'code':
                    target_size = self.chunk_size
                else:
                    target_size = max(1, self.chunk_size - len(prefix))
                saved_size = self._recursive.chunk_size
                saved_overlap = self._recursive.overlap
                self._recursive.configure(chunk_size=target_size, overlap=0)
                try:
                    sub_chunks = self._recursive.split(content)
                finally:
                    self._recursive.configure(chunk_size=saved_size, overlap=saved_overlap)
                for sc in sub_chunks:
                    sc_piece = prefix + sc
                    if len(current) + len(sc_piece) + (1 if current else 0) <= self.chunk_size:
                        current = current + '\n' + sc_piece if current else sc_piece
                    else:
                        if current.strip():
                            chunks.append(current)
                        # 代码切分禁用 char-level overlap，避免拆碎语句；其余类型保留 overlap 衔接
                        # overlap 置于 prefix 之后、sc 之前，保证路径前缀始终在 chunk 头部
                        if self.overlap > 0 and self.doc_type != 'code' and chunks:
                            tail = chunks[-1][-self.overlap:]
                            overlap_piece = prefix + tail + sc
                            if len(overlap_piece) <= self.chunk_size:
                                current = overlap_piece
                            else:
                                current = sc_piece
                        else:
                            current = sc_piece
            current_path = path

        if current.strip():
            chunks.append(current)
        return chunks

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        if self.doc_type == 'markdown':
            units = self._parse_markdown(text)
        elif self.doc_type == 'html':
            units = self._parse_html(text)
        elif self.doc_type == 'docx':
            units = self._parse_docx(text)
        elif self.doc_type == 'code':
            units = self._parse_code(text)
        else:
            raise ValueError(f"不支持的文档类型: {self.doc_type}")
        return self._build_chunks(units)


class SplitterFactory:
    """
    分块策略工厂
    通过策略名（字符串）+ 参数创建对应策略实例，避免客户端直接依赖具体策略类
    支持运行时注册新策略，符合开闭原则
    """

    # 策略注册表：name -> 策略类
    _registry = {
        'fixed_length': FixedLengthChunking,
        'semantic': SemanticChunking,
        'recursive': RecursiveChunking,
        'structural': StructuralChunking,
    }

    @classmethod
    def create_strategy(cls, name: str, **kwargs) -> ChunkingStrategy:
        """
        创建分块策略实例
        :param name: 策略名（不区分大小写）
        :param kwargs: 透传给策略构造函数的参数
        :return: 策略实例
        """
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"未知的分块策略 '{name}'")
        return cls._registry[key](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """返回已注册的所有策略名"""
        return list(cls._registry.keys())


class TextSplitter:
    """
    分块上下文类（Strategy 模式的 Context）
    持有一个 ChunkingStrategy 实例并委托其完成切分，支持运行时切换策略
    内部缓存已使用过的策略实例（按 name 索引），切换时优先复用，避免重复构造
    （如 SemanticChunking 的嵌入模型重复加载）
    """

    def __init__(self, name: str = None, **kwargs):
        """
        通过策略名 + 参数构造，等价于首次 set_strategy
        :param name: 策略名，如 "fixed_length"、"semantic"
        :param kwargs: 透传给策略构造函数的参数
        """
        self._cache: dict[str, ChunkingStrategy] = {}
        # 当前策略
        self.strategy = None
        self.set_strategy(name, **kwargs)

    def set_strategy(self, name: str, **kwargs) -> None:
        """
        切换或初始化分块策略
        缓存命中则复用已有策略实例（并按 kwargs 原地 configure），
        未命中则通过工厂创建并缓存
        :param name: 策略名（不区分大小写）
        :param kwargs: 命中时调用 configure 原地更新；未命中时透传给构造函数
        """
        if name is None:
            return None
        key = name.lower()
        if key in self._cache:
            self.strategy = self._cache[key]
            if kwargs:
                self.strategy.configure(**kwargs)
        else:
            self.strategy = SplitterFactory.create_strategy(name, **kwargs)
            self._cache[key] = self.strategy

    def split(self, text: str) -> list[str]:
        """委托当前策略执行切分"""
        return self.strategy.split(text)


text_splitter = TextSplitter()

if __name__ == '__main__':
    sample = (
        "今天天气真好，阳光明媚。我想出去散步，感受大自然。"
        "量子力学是研究微观粒子运动规律的物理学分支，与相对论并称现代物理两大支柱。"
        "薛定谔方程描述了量子系统的状态随时间的演化。火锅是四川的传统美食，麻辣鲜香，深受大家喜爱。"
    )

    # 1. 定长切分
    text_splitter.set_strategy(ChunkingStrategyEnum.定长切分.value, chunk_size=20, overlap=2)
    print("\n【定长切分】")
    for i, chunk in enumerate(text_splitter.split(sample)):
        print(f"  chunk{i}: {chunk!r}")

    # 2. 语义切分
    text_splitter.set_strategy(name=ChunkingStrategyEnum.语义切分.value)
    print("\n【语义切分】")
    for i, chunk in enumerate(text_splitter.split(sample)):
        print(f"  chunk{i}: {chunk!r}")

    # 3. 递归切分：段落 → 句子 → 标点 → 空格 → 定长，小分片拼接逼近 chunk_size，保留 overlap
    recursive_sample = (
        "人工智能是计算机科学的一个分支，它企图了解智能的实质。"
        "机器学习是人工智能的核心，使计算机能够从数据中学习规律。"
        "深度学习是机器学习的子领域，使用多层神经网络进行特征学习。\n\n"
        "自然语言处理研究人与计算机之间用自然语言进行有效通信的理论方法。"
        "大语言模型通过海量文本预训练，掌握语言统计规律，能完成翻译、问答、摘要等任务。"
        "检索增强生成结合外部知识库与生成模型，缓解幻觉问题。"
    )
    text_splitter.set_strategy(ChunkingStrategyEnum.递归切分.value, chunk_size=60, overlap=20)
    print("\n【递归切分】")
    for i, chunk in enumerate(text_splitter.split(recursive_sample)):
        print(f"  chunk{i}({len(chunk)}字符): {chunk!r}")

    # 4. 结构切分：按文档类型识别结构单元，chunk 头部保留标题路径，结构内递归切分 + overlap
    # 4.1 markdown
    md_sample = (
        "# RAG 引擎\n"
        "RAG 结合检索与生成，缓解大模型幻觉。\n"
        "## 文档解析\n"
        "支持 PDF、Word、Markdown 等格式的解析，提取结构化文本。"
        "解析后的文本按段落、标题、表格等结构单元组织，便于后续切分。\n"
        "## 文档切分\n"
        "切分策略包括定长、语义、递归、结构切分，需根据文档类型选择合适的策略。"
    )
    text_splitter.set_strategy(ChunkingStrategyEnum.结构切分.value, chunk_size=60, overlap=10, doc_type='markdown')
    print("\n【结构切分-markdown】")
    for i, chunk in enumerate(text_splitter.split(md_sample)):
        print(f"  chunk{i}({len(chunk)}字符): {chunk!r}")

    # 4.2 html
    html_sample = (
        "<h1>机器学习入门</h1>"
        "<p>机器学习是让计算机从数据中学习规律的技术，分为监督学习与无监督学习。</p>"
        "<h2>监督学习</h2>"
        "<p>监督学习使用带标签数据训练模型，常见任务有分类与回归。</p>"
        "<h2>无监督学习</h2>"
        "<p>无监督学习从无标签数据中发现模式，如聚类与降维。</p>"
    )
    text_splitter.set_strategy(ChunkingStrategyEnum.结构切分.value, chunk_size=60, overlap=10, doc_type='html')
    print("\n【结构切分-html】")
    for i, chunk in enumerate(text_splitter.split(html_sample)):
        print(f"  chunk{i}({len(chunk)}字符): {chunk!r}")

    # 4.3 docx（解析后以「Heading N: 标题」标记层级）
    docx_sample = (
        "Heading 1: 项目概述\n"
        "本项目实现一个轻量级 RAG 引擎，包含解析、切分、入库、检索、生成模块。\n"
        "Heading 2: 核心模块\n"
        "切分模块支持多种策略，可按文档类型选择最优方案。\n"
        "Heading 2: 扩展模块\n"
        "向量库基于 chromadb，支持持久化与内存两种模式。\n"
    )
    text_splitter.set_strategy(ChunkingStrategyEnum.结构切分.value, chunk_size=60, overlap=10, doc_type='docx')
    print("\n【结构切分-docx】")
    for i, chunk in enumerate(text_splitter.split(docx_sample)):
        print(f"  chunk{i}({len(chunk)}字符): {chunk!r}")

    # 4.4 代码
    code_sample = (
        "class RAGEngine:\n"
        "    def __init__(self, splitter=None):\n"
        "        self.splitter = splitter or TextSplitter()\n\n"
        "    def ingest(self, text):\n"
        "        chunks = self.splitter.split(text)\n"
        "        return chunks\n"
    )
    text_splitter.set_strategy(ChunkingStrategyEnum.结构切分.value, chunk_size=60, overlap=10, doc_type='code')
    print("\n【结构切分-code】")
    for i, chunk in enumerate(text_splitter.split(code_sample)):
        print(f"  chunk{i}({len(chunk)}字符): {chunk!r}")
