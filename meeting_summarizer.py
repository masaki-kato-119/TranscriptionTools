#!/usr/bin/env python3
"""
会議内容の文字起こしファイルをOpenAI APIで要約して議事録を作成するプログラム
"""

import os
import sys
import argparse
import openai
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeetingSummarizer:
    """会議内容の要約を処理するクラス"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        """
        Args:
            api_key: OpenAI APIキー
            model: 使用するモデル名
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        
        logger.info(f"OpenAI APIクライアントを初期化しました（モデル: {model}）")
    
    def read_transcription_file(self, file_path: str) -> str:
        """
        文字起こしファイルを読み込む
        
        Args:
            file_path: 文字起こしファイルのパス
            
        Returns:
            ファイルの内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"文字起こしファイルを読み込みました: {file_path} ({len(content)}文字)")
            return content
            
        except Exception as e:
            raise Exception(f"ファイル読み込みエラー: {e}")
    
    def create_meeting_summary_prompt(self, text: str, target_length: str = "3000字程度") -> str:
        """
        会議要約用のプロンプトを作成
        
        Args:
            text: 元の文字起こしテキスト
            target_length: 目標文字数
            
        Returns:
            プロンプト文字列
        """
        prompt = f"""以下の会議の文字起こし内容を、構造化された議事録として要約してください。

【議事録作成の要件】
- 会議の要点を中心に{target_length}で要約（2000～4000文字程度）
- 以下の構成で詳細に整理してください：
  1. 会議概要
     - 日時・場所・参加者（テキストに記載されている事実のみ）
     - 議題・会議の目的
     - 会議の進行状況
  2. 主要な議題と議論内容
     - 各議題の詳細な議論内容
     - 参加者の意見・提案（テキストに記載されている事実のみ）
     - 議論のポイント・争点
     - 具体的な数値・データ・事例
  3. 決定事項・結論
     - 決定された事項の詳細
     - 決定に至った理由・背景
     - 決定事項の影響・効果
  4. アクションアイテム
     - 具体的なタスク内容
     - 担当者・責任者（テキストに記載されている事実のみ）
     - 期限・マイルストーン
     - 必要なリソース・予算
  5. 次回検討事項
     - 継続検討が必要な項目
     - 次回会議の議題候補
     - 準備すべき資料・情報
- 架空の参加者名や詳細化は一切行わない
- テキストに記載されている事実のみを使用
- 専門用語は分かりやすく説明
- 重要な数値・データ・日付は明確に記載
- 箇条書き・見出し・表を適切に使用して構造化
- 実用的で後から参照しやすい形式

【元の会議文字起こし】
{text}

【議事録】
"""
        return prompt
    
    def summarize_meeting(self, text: str, target_length: str = "3000字程度") -> str:
        """
        会議内容を要約する
        
        Args:
            text: 元の文字起こしテキスト
            target_length: 目標文字数（2000～4000文字程度）
            
        Returns:
            要約された議事録
        """
        try:
            logger.info(f"会議内容の要約を開始（目標文字数: {target_length}）")
            
            # プロンプトを作成
            prompt = self.create_meeting_summary_prompt(text, target_length)
            
            # システムメッセージを設定
            system_message = """あなたは優秀な議事録作成の専門家です。
会議の文字起こし内容を、2000～4000文字程度の詳細で実用的な議事録にまとめてください。

【議事録作成のポイント】
- 会議の全体像を把握できるよう、概要を明確に記載
- 各議題の議論内容を具体的に整理し、参加者の意見も含める
- 決定事項は理由・背景・影響も含めて詳細に記載
- アクションアイテムは担当者・期限・必要なリソースを明確化
- 次回検討事項は継続性を保てるよう具体的に記載
- 専門用語は分かりやすく説明し、数値・日付は正確に記載
- 後から参照しやすいよう、見出し・箇条書き・表を効果的に使用
- 架空の参加者名や詳細化は一切行わず、テキストに記載されている事実のみを使用

関係者が後から確認しやすく、実務で活用できる議事録を作成してください。"""
            
            # OpenAI APIを呼び出し
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,  # 十分な長さの議事録を生成
                temperature=0.3,  # 創造性と一貫性のバランス
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            
            logger.info(f"議事録の作成が完了しました（文字数: {len(summary)}）")
            return summary
            
        except openai.AuthenticationError:
            raise Exception("OpenAI APIキーが無効です。正しいAPIキーを設定してください。")
        except openai.RateLimitError:
            raise Exception("APIレート制限に達しました。しばらく待ってから再試行してください。")
        except openai.APIError as e:
            raise Exception(f"OpenAI APIエラーが発生しました: {e}")
        except Exception as e:
            raise Exception(f"要約処理中に予期せぬエラーが発生しました: {e}")
    
    def save_meeting_summary(self, summary: str, output_path: str, original_file: str) -> None:
        """
        議事録をファイルに保存
        
        Args:
            summary: 作成された議事録
            output_path: 出力ファイルのパス
            original_file: 元のファイル名
        """
        try:
            # 出力ディレクトリの作成
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 議事録を保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# 会議議事録\n\n")
                f.write(f"**元ファイル**: {original_file}\n")
                f.write(f"**作成日時**: {self._get_current_timestamp()}\n")
                f.write(f"**使用モデル**: {self.model}\n")
                f.write(f"**議事録文字数**: {len(summary)}文字\n\n")
                f.write("---\n\n")
                f.write(summary)
            
            logger.info(f"議事録を {output_path} に保存しました")
            
        except Exception as e:
            raise Exception(f"ファイル保存中にエラーが発生しました: {e}")
    
    def _get_current_timestamp(self) -> str:
        """現在のタイムスタンプを取得"""
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    
    def analyze_meeting_content(self, text: str) -> Dict[str, Any]:
        """
        会議内容の基本情報を分析
        
        Args:
            text: 分析対象の文字起こしテキスト
            
        Returns:
            分析結果の辞書
        """
        lines = text.split('\n')
        paragraphs = [line.strip() for line in lines if line.strip()]
        
        # 会議関連キーワードの検出
        meeting_keywords = {
            '決定': text.count('決定'),
            '検討': text.count('検討'),
            '報告': text.count('報告'),
            '議題': text.count('議題'),
            '質問': text.count('質問'),
            '回答': text.count('回答'),
            '次回': text.count('次回'),
            '期限': text.count('期限'),
            '担当': text.count('担当')
        }
        
        return {
            'total_characters': len(text),
            'total_lines': len(lines),
            'non_empty_lines': len(paragraphs),
            'estimated_reading_time': len(text) // 400,  # 1分400文字として概算
            'meeting_keywords': meeting_keywords,
            'has_meeting_content': any(count > 0 for count in meeting_keywords.values())
        }

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="会議内容の文字起こしファイルをOpenAI APIで要約して議事録を作成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python meeting_summarizer.py transcription.txt meeting_summary.md
  python meeting_summarizer.py meeting.txt summary.md --length "3000字程度"
  python meeting_summarizer.py report.txt summary.md --model gpt-4.1 --length "2500字程度"
        """
    )
    
    parser.add_argument(
        "input_file",
        help="入力文字起こしファイルのパス"
    )
    
    parser.add_argument(
        "output_file",
        help="出力議事録ファイルのパス"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="OpenAI APIキー（環境変数OPENAI_API_KEYからも取得可能）"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1",
        choices=["gpt-4.1", "gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        help="使用するOpenAIモデル（デフォルト: gpt-4.1）"
    )
    
    parser.add_argument(
        "--length", "-l",
        default="3000字程度",
        help="目標文字数（デフォルト: 3000字程度、2000～4000文字程度）"
    )
    
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="会議内容の分析情報も表示"
    )
    
    args = parser.parse_args()
    
    # APIキーの取得
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI APIキーが設定されていません")
        logger.error("以下のいずれかの方法で設定してください:")
        logger.error("1. --api-key オプションで指定")
        logger.error("2. 環境変数 OPENAI_API_KEY を設定")
        sys.exit(1)
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.input_file):
        logger.error(f"入力ファイルが存在しません: {args.input_file}")
        sys.exit(1)
    
    try:
        # 議事録作成処理の開始
        logger.info("会議議事録作成プログラムを開始します")
        
        # 文字起こしファイルを読み込み
        summarizer = MeetingSummarizer(api_key, args.model)
        text_content = summarizer.read_transcription_file(args.input_file)
        
        # 会議内容分析（オプション）
        if args.analyze:
            analysis = summarizer.analyze_meeting_content(text_content)
            print("\n" + "="*60)
            print("会議内容分析結果")
            print("="*60)
            print(f"総文字数: {analysis['total_characters']:,}文字")
            print(f"総行数: {analysis['total_lines']:,}行")
            print(f"空行以外の行数: {analysis['non_empty_lines']:,}行")
            print(f"推定読書時間: {analysis['estimated_reading_time']}分")
            print(f"会議関連キーワード:")
            for keyword, count in analysis['meeting_keywords'].items():
                if count > 0:
                    print(f"  {keyword}: {count}回")
            print("="*60 + "\n")
        
        # 議事録作成実行
        print(f"議事録作成を開始します（目標: {args.length}）...")
        summary = summarizer.summarize_meeting(text_content, args.length)
        
        # 議事録結果を表示
        print("\n" + "="*60)
        print("作成された議事録")
        print("="*60)
        print(summary)
        print("="*60)
        print(f"\n議事録文字数: {len(summary)}文字")
        
        # ファイルに保存
        summarizer.save_meeting_summary(summary, args.output_file, args.input_file)
        
        print(f"\n✅ 議事録の作成が完了しました！")
        print(f"結果は {args.output_file} に保存されています")
        
    except KeyboardInterrupt:
        logger.info("ユーザーによって処理が中断されました")
        sys.exit(0)
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
