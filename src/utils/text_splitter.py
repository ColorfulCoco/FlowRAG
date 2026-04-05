# -*- coding: utf-8 -*-
"""
文本分块工具

将长文档按多种策略分割为适合嵌入的文本块，支持固定长度、段落、语义边界、句子级分块。

Author: CongCongTian
"""

import re
from typing import List, Optional
from pathlib import Path
import hashlib

from src.models.schemas import TextChunk


class TextSplitter:
    """通用文本分块器，按字符数分块并支持重叠"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n",
        length_function: Optional[callable] = None,
    ):
        """
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块之间的重叠字符数
            separator: 主分隔符
            length_function: 长度计算函数，默认 len()
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.length_function = length_function or len
        
        self.paragraph_separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
    
    def split_text(self, text: str) -> List[str]:
        """将文本分割为块列表"""
        if not text or not text.strip():
            import logging
            logging.warning("text_splitter: 输入文本为空或仅含空白，返回空列表")
            return []
        
        splits = text.split(self.separator)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            if split_length > self.chunk_size:
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                sub_chunks = self._split_large_text(split)
                chunks.extend(sub_chunks)
            
            elif current_length + split_length + len(self.separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                
                # 保留上一块尾部作为重叠，维持上下文连续性
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, split] if overlap_text else [split]
                current_length = self.length_function(self.separator.join(current_chunk))
            
            else:
                current_chunk.append(split)
                current_length += split_length + (len(self.separator) if len(current_chunk) > 1 else 0)
        
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return [c.strip() for c in chunks if c.strip()]
    
    def _split_large_text(self, text: str) -> List[str]:
        """将超过 chunk_size 的文本块进一步拆分"""
        for sep in self.paragraph_separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) > 1:
                    sub_splitter = TextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separator=sep,
                    )
                    return sub_splitter.split_text(text)
        
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, chunks: List[str]) -> str:
        """从已累积的块中截取尾部作为重叠文本"""
        if not chunks:
            return ""
        
        full_text = self.separator.join(chunks)
        if len(full_text) <= self.chunk_overlap:
            return full_text
        
        return full_text[-self.chunk_overlap:]
    
    def split_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> List[TextChunk]:
        """读取并分割单个文件，返回 TextChunk 列表"""
        path = Path(file_path)
        tg_id = path.stem  # 提取TG标识，如 "TG-01.txt" -> "TG-01"
        
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        
        text_chunks = self.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = self._generate_id(path.name, i)
            chunks.append(TextChunk(
                id=chunk_id,
                text=chunk_text,
                source_file=path.name,
                chunk_index=i,
                metadata={
                    "file_path": str(path),
                    "tg_id": tg_id,  # 记录TG标识，用于后续相似度检索时按TG过滤
                },
            ))
        
        return chunks
    
    def split_folder(
        self,
        folder_path: str,
        pattern: str = "*.txt",
        encoding: str = "utf-8",
    ) -> List[TextChunk]:
        """批量分割文件夹内匹配 pattern 的所有文件"""
        folder = Path(folder_path)
        all_chunks = []
        
        for file_path in sorted(folder.glob(pattern)):
            chunks = self.split_file(str(file_path), encoding)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _generate_id(self, source: str, index: int) -> str:
        """基于来源文件名和块索引生成唯一 ID"""
        content = f"{source}_{index}"
        hash_value = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{source.replace('.', '_')}_{index}_{hash_value}"


class ParagraphSplitter(TextSplitter):
    """按段落分块，每个段落直接作为一个 chunk"""
    
    def __init__(self, min_chunk_length: int = 10):
        """
        Args:
            min_chunk_length: 最小 chunk 长度，低于此值的段落会被过滤
        """
        super().__init__(chunk_size=0, chunk_overlap=0)
        self.min_chunk_length = min_chunk_length
    
    def split_text(self, text: str) -> List[str]:
        """按段落分割文本，每个段落作为一个 chunk"""
        if not text:
            return []
        
        paragraphs = text.split('\n\n')
        
        # 双换行分割不出多段时，退化为单换行分割
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) >= self.min_chunk_length:
                chunks.append(para)
        
        return chunks


class SemanticTextSplitter(TextSplitter):
    """基于标题层级的语义分块器，优先在标题边界处切分"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        super().__init__(chunk_size, chunk_overlap)
        
        self.heading_pattern = re.compile(
            r'^(#{1,6}\s+.+|第[一二三四五六七八九十\d]+[章节、].+|'
            r'\d+\.\s+.+|\d+\.\d+\s+.+|'
            r'[一二三四五六七八九十]+[、.].+)$',
            re.MULTILINE
        )
    
    def split_text(self, text: str) -> List[str]:
        """在标题边界处分割文本，超长段落回退到基础分块"""
        headings = list(self.heading_pattern.finditer(text))
        
        if not headings:
            return super().split_text(text)
        
        sections = []
        prev_end = 0
        
        for heading in headings:
            if heading.start() > prev_end:
                section = text[prev_end:heading.start()].strip()
                if section:
                    sections.append(section)
            prev_end = heading.start()
        
        if prev_end < len(text):
            section = text[prev_end:].strip()
            if section:
                sections.append(section)
        
        chunks = []
        for section in sections:
            if self.length_function(section) > self.chunk_size:
                sub_chunks = super().split_text(section)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)
        
        return chunks


class SentenceSplitter(TextSplitter):
    """按句子分块，每个 chunk 包含若干完整句子，适合细粒度检索"""
    
    def __init__(
        self,
        max_sentences_per_chunk: int = 2,
        min_chunk_length: int = 20,
        max_chunk_length: int = 300,
    ):
        """
        Args:
            max_sentences_per_chunk: 每个 chunk 最多句子数
            min_chunk_length: 低于此长度的 chunk 会合并到相邻块
            max_chunk_length: 超过此长度强制分割
        """
        super().__init__(chunk_size=max_chunk_length, chunk_overlap=0)
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        
        # 正向前瞻排除数字小数点，只匹配真正的句末标点
        self.sentence_endings = re.compile(r'([。！？；\!\?;]|\.(?=\s|$|[^\d\w]))')
        self.numbering_pattern = re.compile(r'^[\d]+[）\)\.\、]')
    
    def split_text(self, text: str) -> List[str]:
        """按句子边界分割文本"""
        if not text:
            return []
        
        paragraphs = text.split('\n')
        
        all_sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            parts = self.sentence_endings.split(para)
            
            # 正则 split 会把捕获组（标点）单独列为元素，需要拼回句尾
            sentences = []
            i = 0
            while i < len(parts):
                sentence = parts[i].strip()
                if i + 1 < len(parts) and self.sentence_endings.match(parts[i + 1]):
                    sentence += parts[i + 1]
                    i += 2
                else:
                    i += 1
                
                if sentence:
                    sentences.append(sentence)
            
            all_sentences.extend(sentences)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in all_sentences:
            sentence_len = len(sentence)
            
            if (len(current_chunk) >= self.max_sentences_per_chunk or 
                (current_length + sentence_len > self.max_chunk_length and current_chunk)):
                chunk_text = ''.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_length:
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append(chunk_text)
            elif chunks:
                # 末尾残余过短，并入前一个 chunk 避免碎片
                chunks[-1] = chunks[-1] + ' ' + chunk_text
        
        return chunks