#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlitを使った議事録RAGシステムのWebアプリケーション
"""

import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from pathlib import Path
import openai
from dotenv import load_dotenv
import numpy as np

# SentenceTransformerをインポート
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# OpenAIライブラリのバージョン情報を表示
try:
    openai_version = openai.__version__
except:
    openai_version = "不明"

# 環境変数の読み込み
load_dotenv()

# OpenAI APIキーの設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



class MeetingRAGSystem:
    """議事録RAGシステムのクラス"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        初期化
        
        Args:
            db_path: ChromaDBの保存パス
        """
        self.db_path = db_path
        self.collection = None # 先に初期化
        self.embedding_model = None # 先に初期化

        # 埋め込みモデルを初期化
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.sidebar.info("✅ 埋め込みモデルをロードしました。")
            except Exception as e:
                st.sidebar.error(f"埋め込みモデルのロードに失敗: {e}")
        else:
            st.sidebar.error("sentence-transformersが未インストールです。")
        # ChromaDBクライアントの初期化
        try:
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # コレクション名
            self.collection_name = "meeting_minutes"
            
            # コレクションの取得
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                st.success(f"ChromaDBコレクション '{self.collection_name}' に接続しました")
            except:
                st.error(f"コレクション '{self.collection_name}' が見つかりません。先に議事録ファイルを処理してください。")
                self.collection = None
                
        except Exception as e:
            st.error(f"ChromaDBの接続に失敗しました: {e}")
            self.collection = None
    
    def search_documents(self, query: str, n_results: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        ドキュメントを検索
        
        Args:
            query: 検索クエリ
            n_results: 取得する結果数
            
        Returns:
            検索結果のリスト
        """
        if not self.collection:
            return []
        if self.embedding_model is None:
            st.error("埋め込みモデルが利用できないため、検索を実行できません。")
            return []

        try:
            # クエリをベクトル化
            query_embedding = self.embedding_model.encode(query)
            # 全件取得
            all_data = self.collection.get(include=["embeddings", "documents", "metadatas"])
            results = []
            for doc, meta, emb in zip(all_data["documents"], all_data["metadatas"], all_data["embeddings"]):
                # コサイン類似度計算
                if emb is not None:
                    sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                else:
                    sim = 0.0
                if sim >= similarity_threshold:
                    results.append({
                        "content": doc,
                        "metadata": meta,
                        "similarity": sim
                    })
            # 類似度で降順ソート
            results.sort(key=lambda x: x["similarity"], reverse=True)
            # 上位n件だけ返す
            return results[:n_results]
        except Exception as e:
            st.error(f"検索中にエラーが発生しました: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        コレクションの情報を取得
        
        Returns:
            コレクション情報
        """
        if not self.collection:
            return {"total_count": 0, "type_counts": {}, "file_counts": {}}
        
        try:
            count = self.collection.count()
            
            # メタデータの統計を取得
            all_data = self.collection.get(include=["embeddings", "documents", "metadatas"])  # 明示的にembeddingsを含める
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
            
        except Exception as e:
            st.error(f"コレクション情報の取得中にエラーが発生しました: {e}")
            return {"total_count": 0, "type_counts": {}, "file_counts": {}}
    
    def generate_ai_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        AIを使って回答を生成
        
        Args:
            query: ユーザーの質問
            context_docs: 関連文書のリスト
            
        Returns:
            AIの回答
        """
        if not OPENAI_API_KEY:
            return "OpenAI APIキーが設定されていません。.envファイルにOPENAI_API_KEYを設定してください。"
        
        try:
            # コンテキストを構築（より構造化された形式）
            context_parts = []
            total_tokens_estimate = 0
            
            for i, doc in enumerate(context_docs):
                metadata = doc['metadata']
                
                # 文書の重要度を計算（類似度ベース）
                similarity = doc.get('similarity', 0.0)
                
                # 類似度が高い文書はより詳細に表示
                if similarity >= 0.7:  # 高類似度（70%以上）
                    context_parts.append(f"""【文書{i+1} - 高関連】
タイプ: {metadata.get('type', 'N/A')}
タイトル: {metadata.get('title', 'N/A')}
サブタイトル: {metadata.get('subtitle', 'N/A')}
関連度: {similarity:.1%}
内容: {doc['content']}
---""")
                elif similarity >= 0.5:  # 中類似度（50-70%）
                    context_parts.append(f"""【文書{i+1} - 中関連】
タイプ: {metadata.get('type', 'N/A')}
タイトル: {metadata.get('title', 'N/A')}
関連度: {similarity:.1%}
内容: {doc['content'][:500]}{'...' if len(doc['content']) > 500 else ''}
---""")
                else:  # 低類似度（30-50%）
                    context_parts.append(f"""【文書{i+1} - 低関連】
タイプ: {metadata.get('type', 'N/A')}
関連度: {similarity:.1%}
内容: {doc['content'][:300]}{'...' if len(doc['content']) > 300 else ''}
---""")
                
                # トークン数の概算（GPT-4は1トークン≈4文字）
                total_tokens_estimate += len(str(context_parts[-1])) // 4
                
                # トークン数が多すぎる場合は警告
                if total_tokens_estimate > 8000:  # GPT-4のコンテキスト制限を考慮
                    st.warning(f"⚠️ コンテキストが長くなりすぎています（推定トークン数: {total_tokens_estimate}）。最初の{i+1}件の文書のみを使用します。")
                    break
            
            context_text = "\n\n".join(context_parts)

            # 改善されたプロンプトを構築
            system_prompt = """あなたは議事録や断片的な知識をもとに、ユーザーの質問に対して
統合的かつ考察を含めた回答を行う専門家です。

# 回答ルール
1. **複数の文書や断片的な情報を総合的に整理し、必要に応じて推論や考察も加えてください**
2. **根拠となる文書や情報源がある場合は必ず明記してください**
3. **情報が不足している場合や推測を含む場合は、その旨を明記してください（例：「議事録からは明確でないが、一般的には…」「断片的な情報から推測すると…」など）**
4. **具体的な情報（日時、決定事項、アクションアイテム、担当者、期限など）があれば必ず記載してください**
5. **関連する他の議題や文脈があれば、積極的に言及してください**
6. **日本語で簡潔かつ論理的に回答してください**

# 回答の構造
1. 質問に対する直接的な回答（考察や推論を含めてOK）
2. 関連する具体的な情報や根拠（文書名や出典、関連度順）
3. 必要に応じて、関連する他の議題や一般的な知見への言及
4. 不明点や推測部分があれば、その旨を明記

議事録の内容（関連度順）：
{context}

ユーザーの質問：{query}

上記の情報をもとに、断片的な知識を統合し、考察も含めて分かりやすく回答してください。
"""

            user_prompt = system_prompt.format(
                context=context_text,
                query=query
            )
            
            # OpenAI APIを呼び出し（meeting_summarizer.pyと完全に同じ方法）
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "あなたは議事録の専門家です。議事録の内容のみに基づいて、正確で分かりやすい回答を提供します。多くの文書を参照して包括的な回答を作成してください。"},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,  # より詳細な回答のため増加
                temperature=0.2,   # より一貫性のある回答のため低下
                top_p=0.9,        # 創造性と正確性のバランス
                presence_penalty=0.1,  # 繰り返しを避ける
                frequency_penalty=0.1  # 多様性を保つ
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI回答の生成中にエラーが発生しました: {e}"


def main():
    """メインアプリケーション"""
    
    # ページ設定
    st.set_page_config(
        page_title="議事録RAGシステム",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # タイトル
    st.title("📋 議事録RAGシステム")
    st.markdown("---")
    
    # サイドバー
    with st.sidebar:
        st.header("🔧 設定")
        
        # DBパスの設定
        db_path = st.text_input(
            "ChromaDBパス",
            value="./chroma_db",
            help="ChromaDBの保存パスを指定してください"
        )
        
        # 検索結果数の設定
        n_results = st.slider(
            "検索結果数",
            min_value=1,
            max_value=100,  # 最大値を100件に増加
            value=50,      # デフォルト値を50件に変更
            help="検索で取得する結果数を指定してください"
        )
        
        # 類似度閾値の設定
        similarity_threshold = st.slider(
            "類似度閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.3,     # デフォルト値を30%に変更
            step=0.05,     # より細かい調整ができるように0.05刻みに変更
            help="この値以上の類似度を持つ検索結果のみをRAGに使用します（推奨: 30%以上）"
        )
        
        # システム情報の表示
        st.header("ℹ️ システム情報")
        
        # OpenAIライブラリ情報
        st.info(f"OpenAIライブラリバージョン: {openai_version}")
        
        # RAGシステムの初期化
        rag_system = MeetingRAGSystem(db_path)
        
        # コレクション情報の表示
        if rag_system.collection:
            info = rag_system.get_collection_info()
            
            st.metric("総データ数", info["total_count"])
           
    
    # メインコンテンツ
    if not rag_system.collection:
        st.warning("ChromaDBに接続できません。先に議事録ファイルを処理してください。")
        st.info("使用方法:")
        st.code("python meeting_rag_processor.py transcription_20250823_153511.md")
        return
    
    st.header("🔍 検索・質問")
    
    # 検索クエリの入力
    search_query = st.text_input(
        "検索クエリまたは質問を入力してください",
        placeholder="例: リスクアセスメントについて教えて",
        help="議事録の内容について質問したり、特定のキーワードで検索したりできます"
    )
    
    # 検索ボタン（常に表示、内容がない場合は無効化）
    search_button_disabled = not search_query or search_query.strip() == ""
    if st.button("🔍 検索・質問", type="primary", disabled=search_button_disabled):
        if search_button_disabled:
            st.warning("検索クエリを入力してください")
        else:
            with st.spinner("検索中..."):
                # ドキュメントを検索
                search_results = rag_system.search_documents(search_query, n_results, similarity_threshold)
    
    # ヘルプテキスト
    if not search_query or search_query.strip() == "":
        st.info("💡 上記のテキストボックスに質問や検索したい内容を入力してください")
    else:
        st.success(f"✅ 検索クエリ: **{search_query}** が入力されました")
    
    # 検索結果の表示
    if search_query and not search_button_disabled:
        # ドキュメントを検索
        search_results = rag_system.search_documents(search_query, n_results, similarity_threshold)
        
        if search_results:
            st.success(f"{len(search_results)}件の関連文書が見つかりました")
            
            # 検索結果が表示されたら自動的にAI回答を生成
            st.markdown("---")
            st.markdown("## 🤖 AI回答")
            
            # 検索結果を類似度でフィルタリング
            filtered_results = [
                result for result in search_results 
                if result['similarity'] >= similarity_threshold
            ]
            
            # フィルタリング結果の表示
            if filtered_results:
                st.success(f"✅ 類似度閾値({similarity_threshold:.1%})を満たす{len(filtered_results)}件の文書でRAGを実行します")
                
                # 除外された結果の情報
                excluded_count = len(search_results) - len(filtered_results)
                if excluded_count > 0:
                    st.info(f"⚠️ 類似度が低いため{excluded_count}件の文書を除外しました")
                
                # AI回答の生成
                with st.spinner("AIが回答を生成中..."):
                    ai_response = rag_system.generate_ai_response(search_query, filtered_results)
                    
                    # 回答の表示
                    st.success("✅ AI回答が生成されました")
                    
                    # 回答を美しく表示（メインコンテンツ）
                    st.markdown("---")
                    st.markdown("### 💬 AIの回答")
                    
                    # 回答をカード形式で表示
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid #1f77b4;
                        margin: 10px 0;
                        color: #333333;
                        font-size: 16px;
                        line-height: 1.6;
                    ">
                        {ai_response.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # コピーボタンと詳細情報
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if st.button("📋 回答をコピー", key="copy_response"):
                            st.write("✅ コピーしました！")
                    
                    with col2:
                        st.info("💡 回答をコピーして他の場所で使用できます")
                    
                    st.markdown("---")
                    
                    # 回答の詳細情報
                    with st.expander("🔍 AI回答の詳細情報", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**❓ 質問:**", search_query)
                            st.write("**📚 使用された文書数:**", len(filtered_results))
                            st.write("**🔍 類似度閾値:**", f"{similarity_threshold:.1%}")
                        
                        with col2:
                            st.write("**⏰ 回答生成日時:**", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            st.write("**🤖 使用モデル:** GPT-4")
                        
                        # 使用された文書の要約
                        st.write("**📖 参考文書:**")
                        for i, result in enumerate(filtered_results):
                            similarity = result['similarity']
                            st.write(f"**📄 文書{i+1}** (関連度: {similarity:.1%})")
                            st.write(f"- 🏷️ タイプ: {result['metadata'].get('type', 'N/A')}")
                            st.write(f"- 📁 ファイル: {result['metadata'].get('file_name', 'N/A')}")
                            if result['metadata'].get('subtitle'):
                                st.write(f"- 📝 サブタイトル: {result['metadata']['subtitle']}")
                            # ↓↓↓ ここを追加 ↓↓↓
                            with st.expander(f"📄 文書{i+1} の内容（クリックで展開）", expanded=False):
                                st.write(result['content'])
                            st.write("---")                            
                        # プロンプト情報
                        st.write("**🔧 プロンプト情報:**")
                        st.info("議事録の専門家として、類似度閾値を満たす関連文書の内容を基に、質問に正確で分かりやすく回答しました。")
            else:
                st.warning(f"⚠️ 類似度閾値({similarity_threshold:.1%})を満たす文書がありません")
                st.info("以下の方法を試してください：")
                st.write("• 類似度閾値を下げる")
                st.write("• 検索クエリをより具体的にする")
                st.write("• 検索結果数を増やす")
        else:
            st.warning("関連する文書が見つかりませんでした。別のキーワードで試してみてください。")
        
    # フッター
    st.markdown("---")
    st.markdown(
        "**議事録RAGシステム** | "
        "ChromaDB + OpenAI + Streamlit | "
        f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
