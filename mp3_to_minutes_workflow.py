#!/usr/bin/env python3
"""
MP3ファイルから議事録作成・RAG登録までの統合ワークフロー
1. MP3ファイルを無音箇所で分割
2. 分割されたWAVファイルを文字起こし
3. 文字起こし結果をOpenAI APIで要約して議事録を作成
4. 議事録をChromaDBにRAG登録
5. 結果を統合して出力
"""

import os
import sys
import time
from pathlib import Path
from mp3_splitter import MP3Splitter
from wav_transcription_faster import FastWAVTranscription
from meeting_summarizer import MeetingSummarizer

# RAG登録用のインポート
try:
    from meeting_rag_processor import MeetingRAGProcessor
    RAG_AVAILABLE = True
except ImportError:
    print("警告: meeting_rag_processor.pyが見つかりません。RAG登録はスキップされます。")
    RAG_AVAILABLE = False

class MP3ToMinutesWorkflow:
    def __init__(self, mp3_file_path: str):
        """
        MP3から議事録作成・RAG登録までの統合ワークフロー
        
        Args:
            mp3_file_path (str): 処理するMP3ファイルのパス
        """
        self.mp3_file_path = mp3_file_path
        self.splitter = None
        self.transcriber = None
        self.summarizer = None
        self.rag_processor = None
        self.output_folder = None
        self.minutes_file_path = None
        
    def run_workflow(self, 
                     split_params: dict = None,
                     transcription_params: dict = None,
                     summary_params: dict = None,
                     enable_rag: bool = True):
        """
        統合ワークフローを実行
        
        Args:
            split_params (dict): MP3分割のパラメータ
            transcription_params (dict): 文字起こしのパラメータ
            summary_params (dict): 議事録作成のパラメータ
            enable_rag (bool): RAG登録を有効にするかどうか
        """
        # デフォルトパラメータ
        if split_params is None:
            split_params = {
                'min_silence_len': 1000,
                'silence_thresh': -40,
                'keep_silence': 100
            }
        
        if transcription_params is None:
            transcription_params = {
                'model_name': 'tiny',
                'output_format': 'markdown',
                'compute_type': 'int8',
                'device': 'auto'
            }
        
        if summary_params is None:
            summary_params = {
                'model': 'gpt-4.1',
                'target_length': '3000字程度'
            }
        
        print("=" * 80)
        print("MP3 → 議事録作成・RAG登録 統合ワークフローを開始します")
        print("=" * 80)
        
        # ステップ1: MP3ファイルの分割
        print("\n【ステップ1】MP3ファイルの分割")
        print("-" * 50)
        
        if not self._split_mp3(split_params):
            print("エラー: MP3ファイルの分割に失敗しました")
            return False
        
        # ステップ2: 分割されたWAVファイルの文字起こし（高速版）
        print("\n【ステップ2】高速文字起こし処理（Faster Whisper + int8量子化）")
        print("-" * 50)
        
        if not self._transcribe_wavs(transcription_params):
            print("エラー: 文字起こし処理に失敗しました")
            return False
        
        # ステップ3: 文字起こし結果の要約と議事録作成
        print("\n【ステップ3】議事録作成")
        print("-" * 50)
        
        if not self._create_meeting_minutes(summary_params):
            print("エラー: 議事録作成に失敗しました")
            return False
        
        # ステップ4: RAG登録（オプション）
        if enable_rag and RAG_AVAILABLE:
            print("\n【ステップ4】RAG登録（ChromaDB）")
            print("-" * 50)
            
            if not self._register_to_rag():
                print("警告: RAG登録に失敗しましたが、ワークフローは続行します")
        elif not RAG_AVAILABLE:
            print("\n【ステップ4】RAG登録（スキップ）")
            print("-" * 50)
            print("meeting_rag_processor.pyが見つからないため、RAG登録をスキップします")
        else:
            print("\n【ステップ4】RAG登録（無効化）")
            print("-" * 50)
            print("RAG登録が無効化されています")
        
        # ステップ5: 結果の統合とサマリー
        print("\n【ステップ5】結果の統合")
        print("-" * 50)
        
        self._create_workflow_summary()
        
        print("\n" + "=" * 80)
        print("統合ワークフローが完了しました！")
        print("=" * 80)
        
        return True
    
    def _split_mp3(self, params: dict) -> bool:
        """MP3ファイルを分割"""
        try:
            # MP3Splitterインスタンスを作成
            self.splitter = MP3Splitter(self.mp3_file_path)
            
            # 分割処理を実行
            success = self.splitter.process(
                min_silence_len=params['min_silence_len'],
                silence_thresh=params['silence_thresh'],
                keep_silence=params['keep_silence']
            )
            
            if success:
                self.output_folder = self.splitter.output_folder
                print(f"分割完了: {self.output_folder}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"エラー: MP3分割処理で例外が発生しました - {e}")
            return False
    
    def _transcribe_wavs(self, params: dict) -> bool:
        """分割されたWAVファイルを文字起こし"""
        try:
            if not self.output_folder:
                print("エラー: 出力フォルダが設定されていません")
                return False
            
            # 出力ファイル名を生成
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = f"workflow_transcription_{timestamp}.{params['output_format']}"
            
            # FastWAVTranscriptionインスタンスを作成（高速版）
            self.transcriber = FastWAVTranscription(
                str(self.output_folder), 
                output_file
            )
            
            # 高速文字起こし処理を実行（int8量子化）
            success = self.transcriber.process(
                model_name=params['model_name'],
                device=params.get('device', 'auto'),
                compute_type=params.get('compute_type', 'int8'),
                output_format=params['output_format']
            )
            
            if success:
                print(f"文字起こし完了: {output_file}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"エラー: 文字起こし処理で例外が発生しました - {e}")
            return False
    
    def _create_meeting_minutes(self, params: dict) -> bool:
        """文字起こし結果を要約して議事録を作成"""
        try:
            if not self.transcriber:
                print("エラー: 文字起こし結果が利用できません")
                return False
            
            # OpenAI APIキーの取得
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("エラー: 環境変数OPENAI_API_KEYが設定されていません")
                print("議事録作成にはOpenAI APIキーが必要です")
                return False
            
            # 文字起こしファイルのパスを取得
            transcription_file = self.transcriber.output_file
            if not os.path.exists(transcription_file):
                print(f"エラー: 文字起こしファイルが見つかりません: {transcription_file}")
                return False
            
            # 議事録出力ファイル名を生成
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            minutes_file = f"meeting_minutes_{timestamp}.md"
            
            # MeetingSummarizerインスタンスを作成
            self.summarizer = MeetingSummarizer(api_key, params['model'])
            
            # 文字起こしファイルを読み込み
            text_content = self.summarizer.read_transcription_file(transcription_file)
            
            # 議事録作成
            summary = self.summarizer.summarize_meeting(text_content, params['target_length'])
            
            # 議事録を保存
            self.summarizer.save_meeting_summary(summary, minutes_file, transcription_file)
            
            # 議事録ファイルのパスを保存（RAG登録用）
            self.minutes_file_path = os.path.join(os.path.dirname(transcription_file), minutes_file)
            
            print(f"議事録作成完了: {minutes_file}")
            return True
                
        except Exception as e:
            print(f"エラー: 議事録作成処理で例外が発生しました - {e}")
            return False
    
    def _register_to_rag(self) -> bool:
        """議事録をChromaDBにRAG登録"""
        try:
            if not self.minutes_file_path or not os.path.exists(self.minutes_file_path):
                print("エラー: 議事録ファイルが見つかりません")
                return False
            
            print(f"議事録ファイルをRAGシステムに登録中: {self.minutes_file_path}")
            
            # MeetingRAGProcessorインスタンスを作成
            self.rag_processor = MeetingRAGProcessor()
            
            # 議事録をRAGシステムに登録
            self.rag_processor.process_file(self.minutes_file_path)
            
            print("✅ RAG登録が完了しました")
                
            # コレクション情報を表示
            collection_info = self.rag_processor.get_collection_info()
            if collection_info:
                print(f"📊 登録された文書数: {collection_info.get('total_count', 'N/A')}")
                print(f"🗄️ コレクション名: {collection_info.get('name', 'N/A')}")
            
            return True
                
        except Exception as e:
            print(f"エラー: RAG登録処理で例外が発生しました - {e}")
            return False
    
    def _create_workflow_summary(self):
        """処理結果のサマリーを作成"""
        try:
            if not self.output_folder or not self.transcriber:
                return
            
            # サマリーファイルを作成
            summary_file = self.output_folder / "workflow_summary.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("MP3 → 議事録作成・RAG登録 ワークフロー実行サマリー\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"元のMP3ファイル: {self.mp3_file_path}\n")
                f.write(f"分割出力フォルダ: {self.output_folder}\n")
                f.write(f"文字起こし出力: {self.transcriber.output_file}\n")
                
                if self.summarizer:
                    f.write(f"議事録出力: {self.minutes_file_path or 'meeting_minutes_*.md'}\n")
                
                if self.rag_processor:
                    f.write("RAG登録: 完了 ✅\n")
                else:
                    f.write("RAG登録: 未実行/失敗 ❌\n")
                
                f.write(f"処理日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
                
                # 分割されたファイル数
                wav_files = list(self.output_folder.glob("*.wav"))
                f.write(f"分割されたWAVファイル数: {len(wav_files)}\n")
                
                # 文字起こし結果の統計
                if hasattr(self.transcriber, 'transcriptions'):
                    f.write(f"文字起こし完了ファイル数: {len(self.transcriber.transcriptions)}\n")
                    
                    # 言語の統計
                    languages = {}
                    for trans in self.transcriber.transcriptions:
                        lang = trans.get('language', 'unknown')
                        languages[lang] = languages.get(lang, 0) + 1
                    
                    f.write("\n検出された言語:\n")
                    for lang, count in languages.items():
                        f.write(f"  {lang}: {count}ファイル\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("ワークフローが正常に完了しました。\n")
                f.write("議事録が作成されました。\n")
                
                if self.rag_processor:
                    f.write("RAGシステムに登録されました。\n")
                    f.write("Streamlitアプリで質問できます: streamlit run streamlit_rag_app.py\n")
                else:
                    f.write("RAG登録は実行されませんでした。\n")
                    f.write("手動で実行する場合: python meeting_rag_processor.py <議事録ファイル>\n")
            
            print(f"サマリーファイルを作成しました: {summary_file}")
            
        except Exception as e:
            print(f"警告: サマリーファイルの作成に失敗しました - {e}")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法: python mp3_to_minutes_workflow.py <MP3ファイルパス> [--no-rag]")
        print("例: python mp3_to_minutes_workflow.py audio.mp3")
        print("例: python mp3_to_minutes_workflow.py audio.mp3 --no-rag  # RAG登録をスキップ")
        print("\n注意: このワークフローにはOpenAI APIキーが必要です")
        print("環境変数OPENAI_API_KEYを設定してください")
        return
    
    mp3_file_path = sys.argv[1]
    
    # RAG登録の有効/無効を確認
    enable_rag = True
    if len(sys.argv) > 2 and sys.argv[2] == "--no-rag":
        enable_rag = False
        print("RAG登録が無効化されています")
    
    # ファイルの存在確認
    if not os.path.exists(mp3_file_path):
        print(f"エラー: ファイルが見つかりません: {mp3_file_path}")
        return
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 環境変数OPENAI_API_KEYが設定されていません")
        print("議事録作成はスキップされ、文字起こしまで実行されます")
    
    # ワークフローを実行
    workflow = MP3ToMinutesWorkflow(mp3_file_path)
    
    # カスタムパラメータを設定（必要に応じて調整）
    split_params = {
        'min_silence_len': 1000,  # 無音判定長さ（ミリ秒）
        'silence_thresh': -40,    # 無音閾値（dB）
        'keep_silence': 100       # 無音保持長さ（ミリ秒）
    }
    
    transcription_params = {
        'model_name': 'tiny',    # Whisperモデル（tiny, base, small, medium, large）
        'output_format': 'markdown'  # 出力形式（text, json, markdown）
    }
    
    summary_params = {
        'model': 'gpt-4.1',         # OpenAIモデル（より高品質な要約のため）
        'target_length': '3000字程度'  # 目標文字数（2000～4000文字程度）
    }
    
    # ワークフローを実行
    success = workflow.run_workflow(split_params, transcription_params, summary_params, enable_rag)
    
    if success:
        print("\n🎉 ワークフローが正常に完了しました！")
        print(f"📁 分割されたファイル: {workflow.output_folder}")
        print(f"📝 文字起こし結果: {workflow.transcriber.output_file}")
        if workflow.summarizer:
            print("📋 議事録が作成されました")
        if workflow.rag_processor:
            print("🗄️ RAGシステムに登録されました")
        
        print("\n📖 次のステップ:")
        print("1. 作成された議事録を確認")
        print("2. 必要に応じて内容を編集・調整")
        if workflow.rag_processor:
            print("3. StreamlitアプリでAI質問: streamlit run streamlit_rag_app.py")
            print("4. 関係者に共有")
        else:
            print("3. 手動でRAG登録: python meeting_rag_processor.py <議事録ファイル>")
            print("4. 関係者に共有")
    else:
        print("\n❌ ワークフローでエラーが発生しました")

if __name__ == "__main__":
    main()
