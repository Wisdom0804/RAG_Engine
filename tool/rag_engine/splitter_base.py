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
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tool.siem_tool.snow_flake import snow_flake


@dataclass
class Chunk:
    """
    切分结果统一数据结构
    - document: chunk 文本
    - metadata: 自定义元数据（chunk_id / parent_id / path 等）
    - embedding: 嵌入向量，由下游嵌入步骤填充，split 阶段为 None
    - parent: 所属父块 Chunk（内存对象引用，非重复存储）；顶层块为 None
    """
    document: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    parent: Optional['Chunk'] = None


class ChunkingStrategy(ABC):
    """分块策略抽象基类，所有具体切分策略均需实现 split 与 configure 方法"""

    @abstractmethod
    def split(self, text: str) -> list[Chunk]:
        """
        切分文本，返回 Chunk 列表（带元数据）
        :param text: 待切分的文本
        :return: Chunk 列表
        """
        pass

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """
        批量设置 __init__ 中的构造参数，未传入的参数保持原值不变
        :param kwargs: 与各策略 __init__ 同名的可选参数
        """
        pass

    @staticmethod
    def _new_chunk(document: str, **metadata) -> Chunk:
        """统一 Chunk 构造点：自动填充雪花 ID 作为 chunk_id"""
        metadata['chunk_id'] = snow_flake.next_id()
        return Chunk(document=document, metadata=metadata)

    @staticmethod
    def _split_keep_separator(text: str, separator: str) -> list[str]:
        """按匹配末尾切片，分隔符留在前片；纯空白并入邻片，不丢失原文。"""
        pieces = []
        start = 0
        for match in re.finditer(separator, text):
            end = match.end()
            piece = text[start:end]
            if piece.strip():
                pieces.append(piece)
                start = end
            elif pieces:
                pieces[-1] += piece
                start = end
        tail = text[start:]
        if tail:
            if pieces and not tail.strip():
                pieces[-1] += tail
            else:
                pieces.append(tail)
        return pieces


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

    def split(self, text: str) -> list[Chunk]:
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        # 步长 = chunk_size - overlap，保证相邻分块共享 overlap 个字符且每片不超限
        step = self.chunk_size - self.overlap
        while start < text_len:
            end = start + self.chunk_size
            chunks.append(self._new_chunk(text[start:end]))
            if end >= text_len:
                break
            start += step
        return chunks


class SemanticChunking(ChunkingStrategy):
    """
    语义切分策略
    不固定分块大小，通过嵌入函数计算相邻句子的相似度，相似度低于阈值时断开形成新分块
    默认嵌入模型 BAAI/bge-small-zh-v1.5（中文小模型，约 95MB），自定义嵌入方法需重写 encode 方法
    通过 max_chunk_size 限制最大分片大小，防止语义高度连贯时产生超大分片，超大分片按三层降级切分
    """

    def __init__(self, embed_fn=None, threshold: float = 0.51, chunk_size: int = 1000):
        """
        :param embed_fn: 嵌入函数/模型，需支持 encode(list[str]) -> ndarray，
                         默认使用 SentenceTransformer("BAAI/bge-small-zh-v1.5")（中文小模型，约 95MB）
        :param threshold: 相似度阈值，低于该值则断开，取值范围 [0, 1]
        :param chunk_size: 单个分块的最大字符长度，超过则按三层降级切分（语义二分 → 弱边界 → 定长兜底）
        """
        if embed_fn is None:
            embed_fn = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.max_chunk_size = chunk_size
        # 缩写/小数白名单占位保护映射，_protect 写入、_restore 读取（按 block 生命周期清理）
        self._protect_map: dict[str, str] = {}
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

    # ---------- 句子切分：强边界 R1（标点优先 + 换行兜底 + 缩写/小数保护） ----------

    # 需要保护的"看起来像句子结束标点、但实际不是"的模式：
    #   小数（3.14）、多字母缩写（U.S.A. / e.g.）、常见单缩写（Dr. / Mr. / etc.）
    _PROTECT_PATTERN = re.compile(
        r'\d+\.\d+'                                  # 小数：3.14 / 0.5
        r'|(?:[A-Za-z]\.){2,}'                       # 多字母缩写：U.S.A. / e.g. / i.e.
        r'|\b(?:Mr|Mrs|Dr|Ms|Prof|Jr|Sr|St|vs|etc)\.'  # 常见单缩写：Dr. / Mr. / etc.
    )

    def _protect(self, text: str) -> str:
        """用占位符保护缩写/小数，防止被强标点误切；占位映射存入 self._protect_map"""
        self._protect_map.clear()
        counter = 0

        def _sub(m):
            nonlocal counter
            placeholder = f'\x00PROT{counter}\x00'
            counter += 1
            self._protect_map[placeholder] = m.group()
            return placeholder

        return self._PROTECT_PATTERN.sub(_sub, text)

    def _restore(self, text: str) -> str:
        """还原 _protect 保护的缩写/小数"""
        for placeholder, original in self._protect_map.items():
            text = text.replace(placeholder, original)
        return text

    @staticmethod
    def _has_strong_punct(text: str) -> bool:
        """判断文本是否含强标点（。！？.!?…）"""
        return bool(re.search(r'[。！？.!?…]', text))

    def _split_sentences(self, text: str) -> list[str]:
        """
        强边界切句（R1）：段落粗切 + 标点优先 + 换行兜底 + 缩写/小数保护
        - 先按段落分隔符 \\n\\s*\\n 取粗块（兼顾多段文章）
        - 粗块内有强标点则按强标点切句，无强标点才按换行切行（兜底无标点片段/标签/key-value）
        - 切分前用占位符保护缩写/小数，切完后还原；保留分隔符和首尾空白
        """
        units = []
        for block in self._split_keep_separator(text, r'\n\s*\n'):
            block = self._protect(block)
            if self._has_strong_punct(block):
                sents = self._split_keep_separator(block, r'(?<=[。！？.!?…])\s*')
            else:
                # 无强标点的短行/标签/key-value，按换行兜底
                sents = self._split_keep_separator(block, '\n')
            units.extend(self._restore(s) for s in sents if s.strip())
        return units

    def _split_weak(self, text: str) -> list[str]:
        """弱边界切子单元：；：，等次级标点 + 换行，供超大块单句降级切分"""
        return self._split_keep_separator(text, r'(?<=[；;，,：:、])\s*|\n+')

    # ---------- 超大分片处理：语义优先二分 + 三层降级 ----------

    def _semantic_bisect(self, sent_range: tuple[int, int],
                         embeddings: np.ndarray, sentences: list[str]) -> list[str]:
        """
        语义优先二分：在 [start, end) 内找最低相似度相邻对切开，递归。
        复用第一遍已算的 embeddings，不重复编码；断点选在语义最弱处，与"语义切分"立意一致。
        终止条件：片段长度 ≤ max_chunk_size 或只剩单句（单句超长交由上层降级）。
        """
        start, end = sent_range
        current_text = ''.join(sentences[start:end])
        # 长度达标 或 只剩单句（单句超长交由 _split_oversized 降级）
        if len(current_text) <= self.max_chunk_size or end - start <= 1:
            return [current_text]
        # 块内相邻对 (i, i+1)，相似度已算
        sims = [(self._cosine_similarity(embeddings[i], embeddings[i + 1]), i)
                for i in range(start, end - 1)]
        # 最低相似度处断
        _, cut = min(sims, key=lambda x: x[0])
        left = self._semantic_bisect((start, cut + 1), embeddings, sentences)
        right = self._semantic_bisect((cut + 1, end), embeddings, sentences)
        return left + right

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _split_oversized(self, text: str, sent_range: tuple[int, int],
                         embeddings: np.ndarray, sentences: list[str]) -> list[str]:
        """
        超大分片三层降级（语义优先二分，复用第一遍 embeddings）：
        1. 语义二分：在块内最低相似度相邻对处切开，递归至 ≤ max_chunk_size 或只剩单句
        2. 弱边界：单句超长时按 ；：， 等次级标点切，贪心拼接逼近 max_chunk_size
        3. 定长兜底：弱边界切不动时按 max_chunk_size 硬切
        """
        pieces = self._semantic_bisect(sent_range, embeddings, sentences)
        result = []
        for piece in pieces:
            if len(piece) <= self.max_chunk_size:
                result.append(piece)
                continue
            # 单句超长 → 弱边界贪心拼接
            sub_units = self._split_weak(piece)
            if len(sub_units) <= 1:
                # 弱边界切不动，定长兜底
                result.extend(piece[i:i + self.max_chunk_size]
                               for i in range(0, len(piece), self.max_chunk_size))
                continue
            current = ''
            for unit in sub_units:
                if len(unit) > self.max_chunk_size:
                    # 子单元仍超长，定长兜底
                    if current:
                        result.append(current)
                        current = ''
                    result.extend(unit[i:i + self.max_chunk_size]
                                   for i in range(0, len(unit), self.max_chunk_size))
                elif len(current) + len(unit) <= self.max_chunk_size:
                    current += unit
                else:
                    if current:
                        result.append(current)
                    current = unit
            if current:
                result.append(current)
        return result

    def split(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []
        # 切分
        sentences = self._split_sentences(text)
        if len(sentences) == 1 and len(text) <= self.max_chunk_size:
            return [self._new_chunk(text)]

        # 批量计算所有句子的嵌入向量
        embeddings = self.embed_fn.encode([s.strip() for s in sentences])

        # 第一遍：相邻句相似度判定，低于阈值则断开；同时记录每个语义块对应的句子索引区间 [start, end)
        # 供 _split_oversized/_semantic_bisect 复用 embeddings，不重复编码
        chunks = []  # list[(text, start_idx, end_idx)]
        current_start = 0
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                # 相似度低于阈值，断开形成新分块
                chunks.append((''.join(current), current_start, i))
                current = [sentences[i]]
                current_start = i
            else:
                current.append(sentences[i])
        if current:
            chunks.append((''.join(current), current_start, len(sentences)))

        # 限制最大分片大小，防止语义连贯产生超大分片；统一包装为 Chunk
        result: list[Chunk] = []
        for chunk_text, start, end in chunks:
            if len(chunk_text) <= self.max_chunk_size:
                result.append(self._new_chunk(chunk_text))
            else:
                pieces = self._split_oversized(chunk_text, (start, end), embeddings, sentences)
                result.extend(self._new_chunk(p) for p in pieces)
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

    def split(self, text: str) -> list[Chunk]:
        if not text:
            return []
        small_chunks = self._recursive_split(text)
        return [self._new_chunk(c) for c in self._merge_chunks(small_chunks)]


class StructuralChunking(ChunkingStrategy):
    """
    结构切分策略
    接收文档解析后的 markdown 文本，识别标题层级与标题下内容块等结构单元，
    在结构内部递归切分，并在 chunk 头部保留标题路径等元数据，相邻分块保留 overlap。
    单个结构单元超长时，在结构内用自实现的递归切分进一步切分（不依赖兄弟策略 RecursiveChunking）。
    """

    # 针对解析 markdown（PDF/Excel/Docx 解析产物：缺标点、多换行）优化的分隔符序列，粒度由粗到细
    # 与 RecursiveChunking.DEFAULT_SEPARATORS 的区别：去空格层、加列表项边界、加表格单元格兜底、标点降级
    STRUCTURAL_SEPARATORS = [
        '\n\n',                                    # 段落（空行分隔）
        r'\n(?=\s*[-*+]\s|\s*\d+[.)]\s)',          # 列表项边界（换行+列表标记）
        '\n',                                      # 换行（markdown 行级单元，含表格行）
        r'(?<=[。！？.!?…])',                       # 句子标点（有则切，兜底）
        r'(?<=[,，;；:：、])',                      # 次级标点（进一步兜底）
        r'\|',                                     # 表格单元格（最后兜底，行超长才用）
        None,                                      # 定长兜底
    ]

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        :param chunk_size: 每个分块的目标/最大字符长度
        :param overlap: 相邻分块间的重叠字符长度
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = list(self.STRUCTURAL_SEPARATORS)
        self._validate()

    def _validate(self) -> None:
        """原子校验 chunk_size 与 overlap"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError(f"overlap 必须在 [0, {self.chunk_size}) 之间")

    def configure(self, *, chunk_size: int = None, overlap: int = None) -> None:
        """
        批量设置构造参数，未传入的不修改
        :param chunk_size: 每个分块的目标/最大字符长度
        :param overlap: 相邻分块间的重叠字符长度
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if overlap is not None:
            self.overlap = overlap
        self._validate()

    # ---------- markdown 结构单元识别 ----------

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

    # ---------- 解析噪声清洗 ----------

    def _clean_noise(self, text: str) -> str:
        """
        清洗解析噪声：NaN 空值、Unnamed 空表头。
        只清空内容不删列：NaN/Unnamed: N 替换为空字符串，保持表格 | 结构完整；
        不清理空行/空列，避免破坏表格结构（空行在段落粗切时自然过滤）。
        """
        text = re.sub(r'\bNaN\b', '', text)           # Excel 空值
        text = re.sub(r'Unnamed:\s*\d+', '', text)    # Excel 空表头
        return text

    # ---------- 自实现递归切分（解耦 RecursiveChunking，分隔符针对解析 markdown） ----------

    def _split_by_separator(self, text: str, separator, chunk_size: int) -> list[str]:
        """按单个分隔符切分文本，保留原始分隔符以便直接拼接"""
        if separator is None:
            # 定长兜底
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        return self._split_keep_separator(text, separator)

    def _recursive_split(self, text: str, sep_idx: int = 0,
                         chunk_size: int = None) -> list[str]:
        """
        按分隔符层级递归切分，直到每片 ≤ chunk_size。
        用 STRUCTURAL_SEPARATORS，针对解析 markdown 优化。
        若当前分隔符切不动（仍只有一个片段且超长），则换更细的分隔符继续。
        """
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        if len(text) <= chunk_size:
            return [text] if text else []
        if sep_idx >= len(self.separators):
            # 全部分隔符都已尝试，定长兜底
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        separator = self.separators[sep_idx]
        pieces = self._split_by_separator(text, separator, chunk_size)
        # 当前分隔符切不动，换更细的分隔符
        if len(pieces) <= 1:
            return self._recursive_split(text, sep_idx + 1, chunk_size)

        result = []
        for piece in pieces:
            if len(piece) > chunk_size:
                # 该片段仍超长，递归用更细的分隔符
                result.extend(self._recursive_split(piece, sep_idx + 1, chunk_size))
            else:
                result.append(piece)
        return result

    def _merge_chunks(self, chunks: list[str], chunk_size: int = None,
                      overlap: int = None) -> list[str]:
        """将小分片拼接以逼近 chunk_size，相邻分块以 overlap 衔接"""
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        overlap = self.overlap if overlap is None else overlap
        if not chunks:
            return []
        merged = []
        current = ''
        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current) + len(chunk) <= chunk_size:
                current += chunk
            else:
                # 当前累积已达上限，收尾并开启新分块
                merged.append(current)
                if overlap > 0:
                    tail = current[-overlap:]
                    # overlap + chunk 仍在上限内则带上 overlap
                    if len(tail) + len(chunk) <= chunk_size:
                        current = tail + chunk
                    else:
                        current = chunk
                else:
                    current = chunk
        if current:
            merged.append(current)
        return merged

    # ---------- chunk 组装 ----------

    def _build_chunks(self, units: list[dict]) -> list[Chunk]:
        """
        在结构内部递归切分并组装 chunk：
        - 在 chunk 头部保留标题路径元数据 [path]，并写入 metadata['path']
        - 单个结构单元超长时用自实现的 _recursive_split + _merge_chunks 在结构内继续切分
        - 相邻分块保留 overlap（取自前一 chunk 的正文尾部）
        """
        chunks: list[Chunk] = []
        current = ''
        # content_path 表示 current 中已累积内容的所属路径，随 current 的整体赋值同步更新
        content_path = None
        # 前一已收尾 chunk 的正文尾部，用于 overlap 衔接
        prev_tail = ''

        for unit in units:
            path = unit['path']
            content = unit['content']
            if not content:
                continue

            # 标题路径变化时，若当前分块非空则收尾
            if path and path != content_path and current.strip():
                chunks.append(self._new_chunk(current, path=content_path))
                prev_tail = current[-self.overlap:] if self.overlap > 0 else ''
                current = ''

            prefix = f"[{path}]\n" if path else ''
            piece = prefix + content

            if len(piece) <= self.chunk_size:
                if len(current) + len(piece) + (1 if current else 0) <= self.chunk_size:
                    if current:
                        current = current + '\n' + piece
                    else:
                        current = piece
                        content_path = path
                else:
                    if current.strip():
                        chunks.append(self._new_chunk(current, path=content_path))
                        prev_tail = current[-self.overlap:] if self.overlap > 0 else ''
                    # overlap 取前一片尾部，置于 prefix 之后、content 之前，保证路径前缀始终在 chunk 头部
                    if self.overlap > 0 and prev_tail:
                        overlap_piece = prefix + prev_tail + content
                        if len(overlap_piece) <= self.chunk_size:
                            current = overlap_piece
                        else:
                            current = piece
                    else:
                        current = piece
                    content_path = path
            else:
                # 单个结构单元超长，在结构内自实现递归切分（不依赖 RecursiveChunking，无临时改配置再还原）
                target_size = max(1, self.chunk_size - len(prefix))
                sub_pieces = self._recursive_split(content, chunk_size=target_size)
                sub_chunks = self._merge_chunks(sub_pieces, chunk_size=target_size, overlap=0)
                for sc_text in sub_chunks:
                    sc_piece = prefix + sc_text
                    if len(current) + len(sc_piece) + (1 if current else 0) <= self.chunk_size:
                        if current:
                            current = current + '\n' + sc_piece
                        else:
                            current = sc_piece
                            content_path = path
                    else:
                        if current.strip():
                            chunks.append(self._new_chunk(current, path=content_path))
                            prev_tail = current[-self.overlap:] if self.overlap > 0 else ''
                        if self.overlap > 0 and prev_tail:
                            overlap_piece = prefix + prev_tail + sc_text
                            if len(overlap_piece) <= self.chunk_size:
                                current = overlap_piece
                            else:
                                current = sc_piece
                        else:
                            current = sc_piece
                        content_path = path

        if current.strip():
            chunks.append(self._new_chunk(current, path=content_path))
        return chunks

    def split(self, text: str) -> list[Chunk]:
        if not text:
            return []
        # 入口处清洗解析噪声（NaN 空值、Unnamed 空表头），_parse_markdown 与切分均用清洗后的文本
        text = self._clean_noise(text)
        units = self._parse_markdown(text)
        return self._build_chunks(units)


class ParentChildChunking(ChunkingStrategy):
    """
    父子切分策略（两阶段切分，依赖注入）

    接收两个已构造好的 ChunkingStrategy 实例：
      - parent_splitter：父块切分器
      - child_splitter：子块切分器
    切分时先将整篇文本切成较大的父块，再在每个父块内切出较小的子块。
    检索时可用子块匹配、用父块补充上下文（Parent Document Retrieval）。

    split() 返回全部子块 Chunk；每个子块通过 .parent 引用所属父块对象，
    并在 metadata['parent_id'] 中记录父块雪花 ID（可序列化跨引用）。
    """

    def __init__(self, parent_splitter: ChunkingStrategy, child_splitter: ChunkingStrategy):
        """
        :param parent_splitter: 父块切分器实例（由外部构造并注入）
        :param child_splitter: 子块切分器实例（由外部构造并注入）
        """
        self.parent_splitter = parent_splitter
        self.child_splitter = child_splitter

    def configure(self, parent_splitter: ChunkingStrategy = None,
                  child_splitter: ChunkingStrategy = None) -> None:
        """
        批量替换切分器实例，未传入的保持原值
        :param parent_splitter: 新的父块切分器实例
        :param child_splitter: 新的子块切分器实例
        """
        if parent_splitter is not None:
            self.parent_splitter = parent_splitter
        if child_splitter is not None:
            self.child_splitter = child_splitter

    def split(self, text: str) -> list[Chunk]:
        """
        先父切分、再对每个父块子切分，返回全部子块 Chunk。
        每个子块：
          - .parent 指向所属父块 Chunk 对象（内存引用，不重复存储）
          - metadata['parent_id'] = 父块雪花 ID（可序列化跨引用）
        """
        if not text:
            return []
        children: list[Chunk] = []
        for parent in self.parent_splitter.split(text):
            parent_id = parent.metadata.get('chunk_id')
            for child in self.child_splitter.split(parent.document):
                child.parent = parent
                child.metadata['parent_id'] = parent_id
                children.append(child)
        return children


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
        'parent_child': ParentChildChunking,
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

    def set_strategy(self, name: str = None, **kwargs) -> None:
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
            return None
        else:
            self.strategy = SplitterFactory.create_strategy(name, **kwargs)
            self._cache[key] = self.strategy
            return None

    def get_cached_strategy(self, name: str) -> Optional[ChunkingStrategy]:
        """
        从缓存取已构造的策略实例，未命中返回 None。
        调用方自负其责：父子同名策略且配置不同时，不要从缓存取，
        应直接通过 SplitterFactory.create_strategy 另建独立实例。
        """
        if not name:
            return None
        return self._cache.get(name.lower())

    def split(self, text: str) -> list[Chunk]:
        """委托当前策略执行切分，返回 Chunk 列表"""
        return self.strategy.split(text)


text_splitter = TextSplitter()
