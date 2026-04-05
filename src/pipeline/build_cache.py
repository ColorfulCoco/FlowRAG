# -*- coding: utf-8 -*-
"""
构建缓存管理器

记录已处理的文件，避免重复构建。

Author: CongCongTian
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class FileRecord:
    """单个文件的处理记录"""
    
    def __init__(
        self,
        file_path: str,
        file_size: int,
        modified_time: float,
        file_hash: str,
        processed_time: str,
        tg_id: str,
        nodes_count: int = 0,
        edges_count: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        self.file_path = file_path
        self.file_size = file_size
        self.modified_time = modified_time
        self.file_hash = file_hash
        self.processed_time = processed_time
        self.tg_id = tg_id
        self.nodes_count = nodes_count
        self.edges_count = edges_count
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "file_path": self.file_path,
            "file_size": self.file_size,
            "modified_time": self.modified_time,
            "file_hash": self.file_hash,
            "processed_time": self.processed_time,
            "tg_id": self.tg_id,
            "nodes_count": self.nodes_count,
            "edges_count": self.edges_count,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileRecord":
        """从字典反序列化"""
        return cls(
            file_path=data["file_path"],
            file_size=data["file_size"],
            modified_time=data["modified_time"],
            file_hash=data["file_hash"],
            processed_time=data["processed_time"],
            tg_id=data["tg_id"],
            nodes_count=data.get("nodes_count", 0),
            edges_count=data.get("edges_count", 0),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


class BuildCache:
    """构建缓存管理器

    跟踪已处理文件和阶段2（关键词、嵌入、挂载）状态，支持增量构建。
    缓存检查不依赖 stage2_cache 标记，而是直接检查文件列表和节点状态。
    """
    
    VERSION = "1.2"  # 升级版本以支持 token 统计
    
    def __init__(self, cache_file: str):
        """初始化缓存管理器"""
        self.cache_file = Path(cache_file)
        self.files: Dict[str, FileRecord] = {}
        self.statistics: Dict[str, Any] = {}
        self.stage2_cache: Dict[str, Any] = {}
        self.token_statistics: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        """从磁盘加载缓存文件"""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            version = data.get("version", "1.0")
            if version not in ["1.0", "1.1", "1.2"]:
                from src.config import LOG_VERBOSE
                if LOG_VERBOSE:
                    print(f"      [缓存] 版本不匹配，重置缓存")
                return
            
            self.files = {
                filename: FileRecord.from_dict(record)
                for filename, record in data.get("files", {}).items()
            }
            
            self.statistics = data.get("statistics", {})
            self.stage2_cache = data.get("stage2_cache", {})
            self.token_statistics = data.get("token_statistics", {})
            
            from src.config import LOG_VERBOSE
            if LOG_VERBOSE:
                print(f"      [缓存] 已加载 {len(self.files)} 条文件记录")
        
        except Exception as e:
            print(f"      [缓存] 加载失败: {e}，将创建新缓存")
            self.files = {}
            self.statistics = {}
            self.stage2_cache = {}
            self.token_statistics = {}
    
    def save(self):
        """持久化缓存到磁盘"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._update_statistics()
        data = {
            "version": self.VERSION,
            "last_update": datetime.now().isoformat(),
            "statistics": self.statistics,
            "files": {
                filename: record.to_dict()
                for filename, record in self.files.items()
            },
            "stage2_cache": self.stage2_cache,
            "token_statistics": self.token_statistics,
        }
        
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _update_statistics(self):
        """更新统计信息"""
        self.statistics = {
            "total_files": len(self.files),
            "successful_files": sum(1 for r in self.files.values() if r.success),
            "failed_files": sum(1 for r in self.files.values() if not r.success),
            "total_nodes": sum(r.nodes_count for r in self.files.values()),
            "total_edges": sum(r.edges_count for r in self.files.values()),
        }
    
    def is_file_cached(self, file_path: Path) -> bool:
        """检查文件是否已缓存且成功处理"""
        filename = file_path.name
        if filename not in self.files:
            return False
        record = self.files[filename]
        if not record.success:
            return False
        return True
    
    def add_file_record(
        self,
        file_path: Path,
        tg_id: str,
        nodes_count: int = 0,
        edges_count: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """添加文件处理记录"""
        filename = file_path.name
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        modified_time = file_stat.st_mtime
        
        record = FileRecord(
            file_path=str(file_path),
            file_size=file_size,
            modified_time=modified_time,
            file_hash="",
            processed_time=datetime.now().isoformat(),
            tg_id=tg_id,
            nodes_count=nodes_count,
            edges_count=edges_count,
            success=success,
            error_message=error_message,
        )
        
        self.files[filename] = record
    
    def get_file_record(self, filename: str) -> Optional[FileRecord]:
        """获取文件记录"""
        return self.files.get(filename)
    
    def clear(self):
        """清除全部缓存并删除缓存文件"""
        self.files = {}
        self.statistics = {}
        self.stage2_cache = {}
        self.token_statistics = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_cached_files(self) -> List[str]:
        """获取所有已缓存的文件名"""
        return list(self.files.keys())
    
    def print_statistics(self):
        """打印统计信息"""
        if not self.statistics:
            self._update_statistics()
        
        print(f"\n      [缓存统计]")
        print(f"        总文件数: {self.statistics.get('total_files', 0)}")
        print(f"        成功: {self.statistics.get('successful_files', 0)}")
        print(f"        失败: {self.statistics.get('failed_files', 0)}")
        print(f"        总节点数: {self.statistics.get('total_nodes', 0)}")
        print(f"        总边数: {self.statistics.get('total_edges', 0)}")
    
    # ========== 阶段2缓存 ==========
    
    @staticmethod
    def compute_documents_hash(folder_path: str, pattern: str = "*.txt") -> str:
        """基于文件名列表计算文件夹哈希（不读取文件内容）"""
        folder = Path(folder_path)
        files = sorted(folder.glob(pattern))
        file_names = [f.name for f in files if f.is_file()]
        combined = "||".join(file_names)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    @staticmethod
    def compute_graph_hash(graph: Any) -> str:
        """基于节点和边 ID 计算图谱哈希"""
        node_ids = sorted(graph.nodes.keys())
        edge_tuples = sorted([(e.source_id, e.target_id, e.relation_name) for e in graph.edges])
        
        content = f"nodes:{','.join(node_ids)}||edges:{str(edge_tuples)}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def should_skip_embeddings_generation(self, folder_path: str, pattern: str = "*.txt") -> bool:
        """检查是否可以跳过节点嵌入生成（residual 依赖邻居上下文，有新文件需全量重建）"""
        folder = Path(folder_path)
        if not folder.exists():
            return False
        
        current_files = [f.name for f in folder.glob(pattern) if f.is_file()]
        
        if not current_files:
            return False
        
        for filename in current_files:
            if filename not in self.files or not self.files[filename].success:
                return False
        
        return True
    
    def should_skip_knowledge_mounting(self, graph: Any) -> bool:
        """检查是否可以跳过知识挂载（直接检查节点 enrichment_chunks 状态）"""
        unmounted_nodes = [
            node_id for node_id, node in graph.nodes.items()
            if len(node.enrichment_chunks) == 0
        ]
        
        if unmounted_nodes:
            return False
        return True
    
    def update_stage2_cache(
        self,
        graph: Optional[Any] = None,
        folder_path: Optional[str] = None,
        pattern: Optional[str] = None,
        keywords_extracted: Optional[bool] = None,
        embeddings_generated: Optional[bool] = None,
        documents_processed: Optional[bool] = None,
        knowledge_mounted: Optional[bool] = None,
    ):
        """更新阶段2缓存状态（传入非 None 的参数会被更新）"""
        if not self.stage2_cache:
            self.stage2_cache = {}
        
        if folder_path is not None and pattern is not None:
            self.stage2_cache["documents_hash"] = self.compute_documents_hash(folder_path, pattern)
        
        if graph is not None:
            self.stage2_cache["graph_hash"] = self.compute_graph_hash(graph)
        
        if documents_processed is not None:
            self.stage2_cache["documents_processed"] = documents_processed
        
        if keywords_extracted is not None:
            self.stage2_cache["keywords_extracted"] = keywords_extracted
        
        if embeddings_generated is not None:
            self.stage2_cache["embeddings_generated"] = embeddings_generated
        
        if knowledge_mounted is not None:
            self.stage2_cache["knowledge_mounted"] = knowledge_mounted
        
        self.stage2_cache["last_update"] = datetime.now().isoformat()
    
    def clear_stage2_cache(self):
        """清除阶段2缓存"""
        self.stage2_cache = {}
    
    # ========== Token 统计 ==========
    
    def get_cumulative_tokens(self) -> Dict[str, int]:
        """获取累计 token 统计"""
        return self.token_statistics.get("cumulative", {
            "graph_generation": 0,
            "graph_generation_prompt": 0,
            "graph_generation_completion": 0,
            "text_embedding": 0,
            "keyword_embedding": 0,
            "node_embedding": 0,
        })
    
    def update_token_statistics(self, token_stats_dict: Dict[str, Any]):
        """更新 token 统计（接收 TokenStatistics.to_dict() 的输出）"""
        self.token_statistics = token_stats_dict
        self.token_statistics["last_update"] = datetime.now().isoformat()