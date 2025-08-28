import os
import sys
import json
import datetime
from pathlib import Path
from faster_whisper import WhisperModel
from typing import List, Dict, Optional
import time

class FastWAVTranscription:
    def __init__(self, input_folder: str, output_file: Optional[str] = None):
        """
        WAVファイルから文字起こしを行うクラス（faster_whisper使用）
        
        Args:
            input_folder (str): WAVファイルが格納されているフォルダのパス
            output_file (str, optional): 出力するテキストファイルのパス
        """
        self.input_folder = Path(input_folder)
        self.output_file = output_file
        self.model = None
        self.transcriptions = []
        
        # 出力ファイルが指定されていない場合、自動生成
        if not self.output_file:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_file = f"transcription_fast_{timestamp}.txt"
        
    def load_whisper_model(self, model_name: str = "base", device: str = "auto", compute_type: str = "int8"):
        """
        Faster Whisperモデルを読み込み（int8量子化で高速化）
        
        Args:
            model_name (str): 使用するWhisperモデル名（tiny, base, small, medium, large）
            device (str): 使用デバイス（"cpu", "cuda", "auto"）
            compute_type (str): 計算精度（"int8", "int16", "float16", "float32"）
        """
        try:
            print(f"Faster Whisperモデル '{model_name}' を読み込み中...")
            print(f"デバイス: {device}, 計算精度: {compute_type}")
            
            start_time = time.time()
            
            # int8量子化でモデルを読み込み
            self.model = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=os.cpu_count(),  # CPUスレッド数を最大に
                num_workers=1  # 並列処理のワーカー数
            )
            
            load_time = time.time() - start_time
            print(f"モデル '{model_name}' の読み込みが完了しました（{load_time:.2f}秒）")
            print(f"int8量子化により高速化されています")
            return True
            
        except Exception as e:
            print(f"エラー: Faster Whisperモデルの読み込みに失敗しました - {e}")
            return False
    
    def get_wav_files(self) -> List[Path]:
        """フォルダ内のWAVファイルを取得（ファイル名順でソート）"""
        wav_files = []
        
        if not self.input_folder.exists():
            print(f"エラー: 入力フォルダが見つかりません: {self.input_folder}")
            return wav_files
        
        # WAVファイルを検索
        for file_path in self.input_folder.glob("*.wav"):
            wav_files.append(file_path)
        
        # ファイル名でソート（chunk_001, chunk_002, ...の順）
        wav_files.sort(key=lambda x: x.name)
        
        print(f"WAVファイルを {len(wav_files)} 個見つけました")
        return wav_files
    
    def transcribe_audio(self, audio_file: Path) -> Dict:
        """
        音声ファイルを文字起こし（faster_whisper使用）
        
        Args:
            audio_file (Path): 文字起こしする音声ファイルのパス
            
        Returns:
            Dict: 文字起こし結果を含む辞書
        """
        try:
            print(f"文字起こし中: {audio_file.name}")
            
            start_time = time.time()
            
            # Faster Whisperで文字起こし
            segments, info = self.model.transcribe(
                str(audio_file),
                beam_size=5,  # ビームサイズ（精度と速度のバランス）
                best_of=5,    # 最良候補数
                temperature=0.0,  # 温度（0.0で決定論的）
                condition_on_previous_text=False,  # 前のテキストに依存しない
                initial_prompt=None,  # 初期プロンプトなし
                word_timestamps=False,  # 単語タイムスタンプ無効（高速化）
                vad_filter=True,  # VADフィルター有効（無音部分をスキップ）
                vad_parameters=dict(min_silence_duration_ms=500)  # 無音判定の閾値
            )
            
            # セグメントからテキストを結合
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            full_text = " ".join(text_parts).strip()
            
            processing_time = time.time() - start_time
            
            # ファイル名から時間情報を抽出
            time_info = self.extract_time_from_filename(audio_file.name)
            
            transcription_data = {
                'file_name': audio_file.name,
                'time_info': time_info,
                'text': full_text,
                'language': info.language,
                'language_probability': info.language_probability,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            print(f"文字起こし完了: {audio_file.name} ({processing_time:.2f}秒)")
            return transcription_data
            
        except Exception as e:
            print(f"エラー: {audio_file.name} の文字起こしに失敗しました - {e}")
            return {
                'file_name': audio_file.name,
                'time_info': self.extract_time_from_filename(audio_file.name),
                'text': f"[文字起こしエラー: {e}]",
                'language': 'unknown',
                'language_probability': 0.0,
                'processing_time_seconds': 0.0,
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def extract_time_from_filename(self, filename: str) -> str:
        """
        ファイル名から時間情報を抽出
        
        Args:
            filename (str): ファイル名
            
        Returns:
            str: 抽出された時間情報
        """
        try:
            # chunk_001_02m30s500ms.wav の形式から時間を抽出
            if 'm' in filename and 's' in filename:
                # 分と秒の部分を抽出
                time_part = filename.split('_')[2].replace('.wav', '')
                return time_part
            else:
                return "不明"
        except:
            return "不明"
    
    def save_transcription(self, output_format: str = "text"):
        """
        文字起こし結果を保存
        
        Args:
            output_format (str): 出力形式（"text", "json", "markdown"）
        """
        if not self.transcriptions:
            print("保存する文字起こし結果がありません")
            return
        
        try:
            if output_format == "text":
                self._save_as_text()
            elif output_format == "json":
                self._save_as_json()
            elif output_format == "markdown":
                self._save_as_markdown()
            else:
                print(f"未対応の出力形式です: {output_format}")
                return
            
            print(f"文字起こし結果を保存しました: {self.output_file}")
            
        except Exception as e:
            print(f"エラー: ファイルの保存に失敗しました - {e}")
    
    def _save_as_text(self):
        """テキスト形式で保存"""
        total_time = sum(t.get('processing_time_seconds', 0) for t in self.transcriptions)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"高速文字起こし結果 - {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write(f"総処理時間: {total_time:.2f}秒\n")
            f.write(f"ファイル数: {len(self.transcriptions)}\n")
            f.write(f"平均処理時間: {total_time/len(self.transcriptions):.2f}秒/ファイル\n")
            f.write("=" * 60 + "\n\n")
            
            for i, trans in enumerate(self.transcriptions, 1):
                f.write(f"[{i:03d}] {trans['file_name']} ({trans['time_info']})\n")
                f.write(f"処理時間: {trans.get('processing_time_seconds', 0):.2f}秒\n")
                f.write(f"言語: {trans.get('language', 'unknown')} (確率: {trans.get('language_probability', 0):.2f})\n")
                f.write("-" * 40 + "\n")
                f.write(trans['text'])
                f.write("\n\n")
    
    def _save_as_json(self):
        """JSON形式で保存"""
        total_time = sum(t.get('processing_time_seconds', 0) for t in self.transcriptions)
        
        output_data = {
            'metadata': {
                'created_at': datetime.datetime.now().isoformat(),
                'input_folder': str(self.input_folder),
                'total_files': len(self.transcriptions),
                'total_processing_time': total_time,
                'average_processing_time': total_time / len(self.transcriptions),
                'model': 'faster-whisper',
                'optimization': 'int8_quantization'
            },
            'transcriptions': self.transcriptions
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def _save_as_markdown(self):
        """Markdown形式で保存"""
        total_time = sum(t.get('processing_time_seconds', 0) for t in self.transcriptions)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 高速文字起こし結果\n\n")
            f.write(f"**作成日時**: {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"**入力フォルダ**: `{self.input_folder}`\n")
            f.write(f"**総ファイル数**: {len(self.transcriptions)}\n")
            f.write(f"**総処理時間**: {total_time:.2f}秒\n")
            f.write(f"**平均処理時間**: {total_time/len(self.transcriptions):.2f}秒/ファイル\n")
            f.write(f"**使用モデル**: Faster Whisper (int8量子化)\n\n")
            
            for i, trans in enumerate(self.transcriptions, 1):
                f.write(f"## {i:03d}. {trans['file_name']} ({trans['time_info']})\n\n")
                f.write(f"**処理時間**: {trans.get('processing_time_seconds', 0):.2f}秒\n")
                f.write(f"**言語**: {trans.get('language', 'unknown')} (確率: {trans.get('language_probability', 0):.2f})\n")
                f.write(f"**処理時刻**: {trans['timestamp']}\n\n")
                f.write("```\n")
                f.write(trans['text'])
                f.write("\n```\n\n")
    
    def process(self, model_name: str = "base", device: str = "auto", compute_type: str = "int8", output_format: str = "text"):
        """
        メイン処理を実行
        
        Args:
            model_name (str): 使用するWhisperモデル名
            device (str): 使用デバイス
            compute_type (str): 計算精度
            output_format (str): 出力形式
        """
        print("=" * 60)
        print("高速WAVファイル文字起こし処理を開始します")
        print(f"モデル: {model_name}, デバイス: {device}, 精度: {compute_type}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Faster Whisperモデルを読み込み
        if not self.load_whisper_model(model_name, device, compute_type):
            return False
        
        # 2. WAVファイルを取得
        wav_files = self.get_wav_files()
        if not wav_files:
            print("処理するWAVファイルが見つかりません")
            return False
        
        # 3. 各ファイルを文字起こし
        print(f"\n{len(wav_files)}個のファイルの文字起こしを開始します...")
        
        for i, wav_file in enumerate(wav_files, 1):
            print(f"\n[{i}/{len(wav_files)}] 処理中...")
            transcription = self.transcribe_audio(wav_file)
            self.transcriptions.append(transcription)
        
        # 4. 結果を保存
        print(f"\n文字起こし結果を保存中...")
        self.save_transcription(output_format)
        
        total_time = time.time() - start_time
        
        print("=" * 60)
        print("処理完了!")
        print(f"処理したファイル数: {len(self.transcriptions)}")
        print(f"総処理時間: {total_time:.2f}秒")
        print(f"平均処理時間: {total_time/len(self.transcriptions):.2f}秒/ファイル")
        print(f"出力ファイル: {self.output_file}")
        print("=" * 60)
        
        return True

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法: python wav_transcription_faster.py <WAVファイルフォルダ> [モデル名] [出力形式] [出力ファイル名]")
        print("モデル名: tiny, base, small, medium, large (デフォルト: base)")
        print("出力形式: text (デフォルト), json, markdown")
        print("例: python wav_transcription_faster.py split_20241201_143022 base markdown")
        return
    
    input_folder = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"
    output_format = sys.argv[3] if len(sys.argv) > 3 else "text"
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    # フォルダの存在確認
    if not os.path.exists(input_folder):
        print(f"エラー: フォルダが見つかりません: {input_folder}")
        return
    
    # 出力形式の検証
    valid_formats = ["text", "json", "markdown"]
    if output_format not in valid_formats:
        print(f"エラー: 無効な出力形式です: {output_format}")
        print(f"有効な形式: {', '.join(valid_formats)}")
        return
    
    # FastWAVTranscriptionインスタンスを作成して処理を実行
    transcriber = FastWAVTranscription(input_folder, output_file)
    
    # int8量子化で高速化
    success = transcriber.process(
        model_name=model_name,
        device="auto",  # 自動検出
        compute_type="int8",  # int8量子化で高速化
        output_format=output_format
    )
    
    if success:
        print("高速文字起こし処理が正常に完了しました")
    else:
        print("高速文字起こし処理に失敗しました")

if __name__ == "__main__":
    main()
