#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
議事録ファイルを断片化してChromaDBに保存するプログラム

依存関係:
- chromadb
- sentence-transformers (推奨、より高品質な埋め込みベクトルを生成)

インストール:
pip install sentence-transformers
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions
import markdown
from pathlib import Path
import numpy as np

# sentence-transformersを直接インポート
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformersがインストールされていません")


class MeetingRAGProcessor:
    """議事録を断片化してChromaDBに保存するクラス"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        初期化
        
        Args:
            db_path: ChromaDBの保存パス
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # コレクション名
        self.collection_name = "meeting_minutes"
        
        # 埋め込みモデルの初期化
        self.embedding_model = None
        self.embedding_dimension = 384  # デフォルト次元数

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformersがインストールされていません。RAG登録を中止します。")

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 SentenceTransformerモデルを初期化中...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
                print(f"✅ SentenceTransformerモデルの初期化完了（次元数: {self.embedding_dimension}）")
            except Exception as e:
                print(f"⚠️  SentenceTransformerの初期化に失敗: {e}")
                print("🔄 デフォルトの埋め込み関数を使用します")
        else:
            print("🔄 デフォルトの埋め込み関数を使用します")
        
        # コレクションの取得または作成
        self._setup_collection()
    
    def _setup_collection(self):
        """コレクションの設定を行う"""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"既存のコレクション '{self.collection_name}' を取得しました")
            if not self._verify_collection_embeddings():
                raise RuntimeError("既存コレクションの埋め込みベクトル次元数が不一致です。安全のため登録を中止します。")
        except chromadb.errors.NotFoundError:
            print(f"🔄 新しいコレクションを作成中...")
            self._create_new_collection()
        # それ以外の例外は再スロー（新規作成しない）
        except Exception as e:
            raise
    
    def _verify_collection_embeddings(self) -> bool:
        """コレクションの埋め込みベクトルが正常に動作するか確認"""
        try:
            if self.collection.count() == 0:
                print("✅ コレクションが空です（新規作成直後）。")
                return True
            test_text = "テスト"
            test_embedding = self.embedding_model.encode(test_text).tolist()
            test_result = self.collection.query(
                query_embeddings=[test_embedding],
                n_results=1,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            #print("DEBUG: test_result['embeddings'] =", test_result["embeddings"])
            if (test_result and 
                "embeddings" in test_result and 
                test_result["embeddings"] and 
                len(test_result["embeddings"]) > 0):
                first_embedding = test_result["embeddings"][0]
                #print("DEBUG: first_embedding =", first_embedding)
                # NumPy配列対応
                if hasattr(first_embedding, "shape"):
                    # 2次元配列 (1, 384) の場合
                    if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                        actual_dim = first_embedding.shape[1]
                        print(f"DEBUG: Detected numpy 2D array, shape={first_embedding.shape}, using shape[1]={actual_dim}")
                    else:
                        actual_dim = first_embedding.shape[0]
                        print(f"DEBUG: Detected numpy array, shape={first_embedding.shape}, using shape[0]={actual_dim}")
                elif isinstance(first_embedding, list):
                    # リストのリストの場合
                    if len(first_embedding) == 1 and isinstance(first_embedding[0], (list, np.ndarray)):
                        actual_dim = len(first_embedding[0])
                        print(f"DEBUG: Detected list of list, using len(first_embedding[0])={actual_dim}")
                    else:
                        actual_dim = len(first_embedding)
                        print(f"DEBUG: Detected list, using len(first_embedding)={actual_dim}")
                else:
                    actual_dim = len(first_embedding)
                    print(f"DEBUG: Fallback, using len(first_embedding)={actual_dim}")

                if actual_dim == self.embedding_dimension:
                    print(f"✅ 埋め込みベクトルの次元数が一致しています（{actual_dim}）")
                    return True
                else:
                    print(f"⚠️  埋め込みベクトルの次元数が不一致: 期待値={self.embedding_dimension}, 実際={actual_dim}")
                    return False
            else:
                print("✅ コレクションが空です（新規作成直後）。")
                return True
        except Exception as e:
            print(f"⚠️  コレクションの検証中にエラー: {e}")
            return False
    
    def _recreate_collection(self):
        """コレクションを再作成する"""
        try:
            print("🔄 既存のコレクションを削除中...")
            self.client.delete_collection(name=self.collection_name)
            print("✅ 既存のコレクションを削除しました")
            self._create_new_collection()
        except Exception as e:
            print(f"❌ コレクションの削除に失敗: {e}")
            raise e
    
    def _create_new_collection(self):
        """新しいコレクションを作成する"""
        try:
            print("🔄 新しいコレクションを作成中...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "会議議事録の断片化データ",
                    "embedding_dimension": self.embedding_dimension
                }
            )
            print(f"✅ 新しいコレクション '{self.collection_name}' を作成しました")
        except Exception as e:
            print(f"❌ コレクションの作成に失敗: {e}")
            raise e
    
    def parse_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """
        Markdownファイルを解析して構造化データに変換
        
        Args:
            file_path: Markdownファイルのパス
            
        Returns:
            解析結果の辞書
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ファイル名から基本情報を抽出
        file_name = Path(file_path).stem
        
        # 議事録の基本構造を解析
        parsed_data = {
            "file_name": file_name,
            "raw_content": content,
            "sections": [],
            "metadata": {}
        }
        
        # ヘッダー情報を抽出
        header_match = re.search(r'# 会議議事録\s*\n\n(.*?)\n\n---', content, re.DOTALL)
        if header_match:
            header_text = header_match.group(1)
            # メタデータを抽出
            metadata = self._extract_metadata(header_text)
            parsed_data["metadata"] = metadata
        
        # セクションを分割
        sections = self._split_into_sections(content)
        parsed_data["sections"] = sections
        
        return parsed_data
    
    def _extract_metadata(self, header_text: str) -> Dict[str, Any]:
        """
        ヘッダーからメタデータを抽出
        
        Args:
            header_text: ヘッダーテキスト
            
        Returns:
            メタデータの辞書
        """
        metadata = {}
        
        # 元ファイル
        if match := re.search(r'\*\*元ファイル\*\*: (.+)', header_text):
            metadata["source_file"] = match.group(1).strip()
        
        # 作成日時
        if match := re.search(r'\*\*作成日時\*\*: (.+)', header_text):
            date_str = match.group(1).strip()
            metadata["created_date"] = date_str
        
        # 使用モデル
        if match := re.search(r'\*\*使用モデル\*\*: (.+)', header_text):
            metadata["model_used"] = match.group(1).strip()
        
        # 議事録文字数
        if match := re.search(r'\*\*議事録文字数\*\*: (\d+)文字', header_text):
            metadata["character_count"] = int(match.group(1))
        
        return metadata
    
    def _split_into_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        コンテンツをセクションに分割
        
        Args:
            content: 議事録の内容
            
        Returns:
            セクションのリスト
        """
        sections = []
        
        # セクションのパターンを定義
        section_patterns = [
            (r'# 1\. 会議概要\s*\n(.*?)(?=\n# 2\.|\n---)', "meeting_summary", "会議概要"),
            (r'# 2\. 主要な議題と議論内容\s*\n(.*?)(?=\n# 3\.|\n---)', "discussion_topics", "主要な議題と議論内容"),
            (r'# 3\. 決定事項・結論\s*\n(.*?)(?=\n# 4\.|\n---)', "decisions", "決定事項・結論"),
            (r'# 4\. アクションアイテム\s*\n(.*?)(?=\n# 5\.|\n---)', "action_items", "アクションアイテム"),
            (r'# 5\. 次回検討事項\s*\n(.*?)(?=\n---|\Z)', "next_agenda", "次回検討事項")
        ]
        
        for pattern, section_type, section_title in section_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                section_content = match.group(1).strip()
                
                # サブセクションにさらに分割
                subsections = self._split_subsections(section_content, section_type)
                
                for i, subsection in enumerate(subsections):
                    sections.append({
                        "type": section_type,
                        "title": section_title,
                        "subtitle": subsection.get("subtitle", ""),
                        "content": subsection["content"],
                        "content_type": subsection.get("content_type", "text")
                    })
        
        return sections
    
    def _split_subsections(self, content: str, section_type: str) -> List[Dict[str, Any]]:
        """
        セクションをサブセクションに分割
        
        Args:
            content: セクションテキスト
            section_type: セクションタイプ
            
        Returns:
            サブセクションのリスト
        """
        subsections = []
        
        if section_type == "discussion_topics":
            # 主要な議題は2.1, 2.2, 2.3で分割
            topic_pattern = r'## (2\.\d+ [^\n]+)\s*\n(.*?)(?=\n## 2\.\d+|\Z)'
            matches = re.findall(topic_pattern, content, re.DOTALL)
            
            for title, topic_content in matches:
                subsections.append({
                    "subtitle": title.strip(),
                    "content": topic_content.strip(),
                    "content_type": "discussion_topic"
                })
        
        elif section_type == "action_items":
            # アクションアイテムはテーブル形式なので、行ごとに分割
            lines = content.split('\n')
            table_started = False
            current_items = []
            
            for line in lines:
                if '|' in line and '---' not in line:
                    if not table_started:
                        table_started = True
                        continue
                    
                    # テーブル行を解析
                    if line.strip() and not line.startswith('|'):
                        continue
                    
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if len(cells) >= 4:
                        task_content = cells[0]
                        if task_content and task_content != "（記載なし）":
                            subsections.append({
                                "subtitle": f"アクション: {task_content}",
                                "content": f"タスク: {task_content}\n担当者: {cells[1]}\n期限: {cells[2]}\nリソース: {cells[3]}",
                                "content_type": "action_item"
                            })
        
        else:
            # その他のセクションはそのまま
            subsections.append({
                "subtitle": "",
                "content": content,
                "content_type": "text"
            })
        
        return subsections
    
    def create_chunks(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析されたデータをチャンクに分割
        
        Args:
            parsed_data: 解析されたデータ
            
        Returns:
            チャンクのリスト
        """
        chunks = []
        chunk_id = 0
        file_name = parsed_data["file_name"]        
        # 基本情報チャンク
        if parsed_data["metadata"]:
            chunks.append({
                "id": f"{file_name}_chunk_{chunk_id:04d}",
                "content": f"会議議事録: {parsed_data['file_name']}\n\n基本情報:\n" + 
                          "\n".join([f"{k}: {v}" for k, v in parsed_data["metadata"].items()]),
                "metadata": {
                    "type": "meeting_info",
                    "file_name": parsed_data["file_name"],
                    "chunk_type": "header",
                    **parsed_data["metadata"]
                }
            })
            chunk_id += 1
        
        # セクションチャンク
        for section in parsed_data["sections"]:
            chunks.append({
                "id": f"{file_name}_chunk_{chunk_id:04d}",
                "content": f"{section['title']}\n\n{section['subtitle']}\n\n{section['content']}",
                "metadata": {
                    "type": section["type"],
                    "title": section["title"],
                    "subtitle": section["subtitle"],
                    "content_type": section["content_type"],
                    "file_name": parsed_data["file_name"],
                    "chunk_type": "section"
                }
            })
            chunk_id += 1
        
        return chunks
    
    def add_to_chromadb(self, chunks: List[Dict[str, Any]]) -> None:
        """
        チャンクをChromaDBに追加
        
        Args:
            chunks: チャンクのリスト
        """
        if not chunks:
            print("追加するチャンクがありません")
            return

        if not self.embedding_model:
            raise RuntimeError("埋め込みモデルが利用できません。登録を中止します。")

        # 既存のデータを確認
        existing_count = self.collection.count()
        print(f"既存のデータ数: {existing_count}")
        
        # 新しいチャンクを追加
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        print("🔄 埋め込みベクトルを生成中...")
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk["content"])
            metadatas.append(chunk["metadata"])
            ids.append(chunk["id"])
            
            # 埋め込みベクトルを生成
            try:
                embedding = self.embedding_model.encode(chunk["content"]).tolist()
                if len(embedding) != self.embedding_dimension:
                    raise ValueError(f"チャンク{i+1}の埋め込みベクトルの次元数が不正: {len(embedding)} (期待値: {self.embedding_dimension})")
                embeddings.append(embedding)
                if (i + 1) % 5 == 0:
                    print(f"  📊 {i + 1}/{len(chunks)}個の埋め込みベクトルを生成完了")
            except Exception as e:
                raise RuntimeError(f"チャンク{i+1}の埋め込みベクトル生成に失敗: {e}")

        print(f"✅ {len(chunks)}個の埋め込みベクトルの生成完了")
        
        # ChromaDBに追加
        try:
            print("🔄 ChromaDBにデータを追加中...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            print(f"✅ {len(chunks)}個のチャンクをChromaDBに追加しました")
            print(f"現在の総データ数: {self.collection.count()}")

        except Exception as e:
            print(f"❌ ChromaDBへの追加中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            # エラーの詳細を確認
            print("\n🔍 エラーの詳細分析:")
            print(f"  チャンク数: {len(chunks)}")
            print(f"  埋め込みベクトル数: {len(embeddings)}")
            print(f"  期待される次元数: {self.embedding_dimension}")
            
            if embeddings:
                print(f"  最初の埋め込みベクトルの次元数: {len(embeddings[0])}")
                print(f"  最後の埋め込みベクトルの次元数: {len(embeddings[-1])}")
            
            raise e
    
    def process_file(self, file_path: str) -> None:
        """
        ファイルを処理してChromaDBに保存
        
        Args:
            file_path: 処理するファイルのパス
        """
        print(f"ファイルを処理中: {file_path}")
        
        # ファイルの存在確認
        if not os.path.exists(file_path):
            print(f"エラー: ファイルが見つかりません: {file_path}")
            return
        
        # Markdownファイルを解析
        parsed_data = self.parse_markdown_file(file_path)
        print(f"解析完了: {len(parsed_data['sections'])}個のセクションを検出")
        
        # チャンクに分割
        chunks = self.create_chunks(parsed_data)
        print(f"チャンク分割完了: {len(chunks)}個のチャンクを作成")
        
        # ChromaDBに保存
        self.add_to_chromadb(chunks)
        
        print("処理完了!")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        クエリで検索
        
        Args:
            query: 検索クエリ
            n_results: 取得する結果数
            
        Returns:
            検索結果のリスト
        """
        try:
            # 埋め込みモデルがある場合は、クエリの埋め込みベクトルを生成
            if self.embedding_model:
                query_embedding = self.embedding_model.encode(query).tolist()
                
                # 埋め込みベクトルの次元数を確認
                if len(query_embedding) != self.embedding_dimension:
                    print(f"⚠️  クエリの埋め込みベクトルの次元数が不正: {len(query_embedding)} (期待値: {self.embedding_dimension})")
                    # 次元数が合わない場合はゼロベクトルを使用
                    query_embedding = [0.0] * self.embedding_dimension
                
                # 埋め込みベクトルを使用して検索
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["embeddings", "documents", "metadatas", "distances"]
                )
            else:
                # 埋め込みモデルがない場合は、テキストクエリを使用
                print("⚠️  埋め込みモデルが利用できないため、テキストクエリを使用します")
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["embeddings", "documents", "metadatas", "distances"]
                )
            
            return results
            
        except Exception as e:
            print(f"❌ 検索中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            # エラーの場合は空の結果を返す
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "embeddings": [[]]
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        コレクションの情報を取得
        
        Returns:
            コレクション情報
        """
        count = self.collection.count()
        
        # メタデータの統計を取得
        all_data = self.collection.get()
        type_counts = {}
        file_counts = {}
        
        for metadata in all_data["metadatas"]:
            if metadata:
                # タイプ別カウント
                chunk_type = metadata.get("chunk_type", "unknown")
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
                
                # ファイル別カウント
                file_name = metadata.get("file_name", "unknown")
                file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        return {
            "total_count": count,
            "type_counts": type_counts,
            "file_counts": file_counts
        }


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="議事録ファイルをChromaDBに保存するプログラム")
    parser.add_argument("file_path", nargs='?', help="処理するMarkdownファイルのパス")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDBの保存パス")
    parser.add_argument("--search", help="検索クエリを実行")
    parser.add_argument("--info", action="store_true", help="コレクション情報を表示")
    
    args = parser.parse_args()
    
    # プロセッサーを初期化
    processor = MeetingRAGProcessor(args.db_path)
    
    if args.info:
        # コレクション情報を表示
        info = processor.get_collection_info()
        print("=== コレクション情報 ===")
        print(f"総データ数: {info['total_count']}")
        print("\nタイプ別カウント:")
        for chunk_type, count in info['type_counts'].items():
            print(f"  {chunk_type}: {count}")
        print("\nファイル別カウント:")
        for file_name, count in info['file_counts'].items():
            print(f"  {file_name}: {count}")
        return
    
    if args.search:
        # 検索を実行
        print(f"検索クエリ: {args.search}")
        results = processor.search(args.search)
        
        print("\n=== 検索結果 ===")
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        )):
            print(f"\n--- 結果 {i+1} (距離: {distance:.4f}) ---")
            print(f"タイプ: {metadata.get('type', 'unknown')}")
            print(f"タイトル: {metadata.get('title', 'N/A')}")
            if metadata.get('subtitle'):
                print(f"サブタイトル: {metadata['subtitle']}")
            print(f"ファイル: {metadata.get('file_name', 'N/A')}")
            print(f"内容: {doc[:200]}...")
        return
    
    # ファイルパスが指定されていない場合
    if not args.file_path:
        print("使用方法:")
        print("  ファイル処理: python meeting_rag_processor.py <ファイルパス>")
        print("  検索実行: python meeting_rag_processor.py --search <検索クエリ>")
        print("  情報表示: python meeting_rag_processor.py --info")
        print("  ヘルプ: python meeting_rag_processor.py --help")
        return
    
    # ファイルを処理
    processor.process_file(args.file_path)


if __name__ == "__main__":
    main()
