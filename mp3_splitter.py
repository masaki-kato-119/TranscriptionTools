import os
import sys
import wave
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import datetime
from pathlib import Path

class MP3Splitter:
    def __init__(self, mp3_file_path):
        """
        MP3ファイルを無音箇所で分割するクラス
        
        Args:
            mp3_file_path (str): 分割するMP3ファイルのパス
        """
        self.mp3_file_path = mp3_file_path
        self.audio = None
        self.output_folder = None
        
    def load_audio(self):
        """MP3ファイルを読み込み"""
        try:
            print(f"MP3ファイルを読み込み中: {self.mp3_file_path}")
            self.audio = AudioSegment.from_mp3(self.mp3_file_path)
            print(f"読み込み完了: 長さ {len(self.audio) / 1000:.2f}秒")
            return True
        except Exception as e:
            print(f"エラー: MP3ファイルの読み込みに失敗しました - {e}")
            return False
    
    def create_output_folder(self):
        """出力フォルダを作成（MP3ファイル名を使用）"""
        try:
            # MP3ファイルのファイル名（拡張子なし）を取得
            mp3_filename = Path(self.mp3_file_path).stem
            
            # フォルダ名を作成（MP3ファイル名を使用）
            folder_name = f"split_{mp3_filename}"
            
            # 現在のディレクトリに作成
            self.output_folder = Path(folder_name)
            self.output_folder.mkdir(exist_ok=True)
            
            print(f"出力フォルダを作成: {self.output_folder}")
            return True
        except Exception as e:
            print(f"エラー: 出力フォルダの作成に失敗しました - {e}")
            return False
    
    def split_audio(self, min_silence_len=1000, silence_thresh=-40, keep_silence=100):
        """
        無音箇所で音声を分割
        
        Args:
            min_silence_len (int): 無音と判定する最小長さ（ミリ秒）
            silence_thresh (int): 無音の閾値（dB）
            keep_silence (int): 分割後に残す無音の長さ（ミリ秒）
        """
        if self.audio is None:
            print("エラー: 音声ファイルが読み込まれていません")
            return []
        
        try:
            print("無音箇所で音声を分割中...")
            print(f"無音判定: 長さ{min_silence_len}ms以上, 閾値{silence_thresh}dB")
            
            # 無音箇所で分割
            audio_chunks = split_on_silence(
                self.audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
            
            print(f"分割完了: {len(audio_chunks)}個のチャンクに分割されました")
            return audio_chunks
            
        except Exception as e:
            print(f"エラー: 音声の分割に失敗しました - {e}")
            return []
    
    def save_chunks(self, audio_chunks):
        """分割された音声チャンクをWAVファイルとして保存"""
        if not audio_chunks:
            print("保存する音声チャンクがありません")
            return []
        
        if not self.output_folder:
            print("エラー: 出力フォルダが作成されていません")
            return []
        
        saved_files = []
        
        for i, chunk in enumerate(audio_chunks):
            try:
                # チャンクの長さを秒で計算
                duration_seconds = len(chunk) / 1000
                
                # ファイル名を作成（録音時間を含む）
                # 時間:分:秒.ミリ秒の形式
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                milliseconds = int((duration_seconds % 1) * 1000)
                
                if minutes > 0:
                    filename = f"chunk_{i+1:03d}_{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms.wav"
                else:
                    filename = f"chunk_{i+1:03d}_{seconds:02d}s{milliseconds:03d}ms.wav"
                
                file_path = self.output_folder / filename
                
                # WAVファイルとして保存
                chunk.export(str(file_path), format="wav")
                
                print(f"保存完了: {filename} ({duration_seconds:.3f}秒)")
                saved_files.append(str(file_path))
                
            except Exception as e:
                print(f"エラー: チャンク{i+1}の保存に失敗しました - {e}")
        
        return saved_files
    
    def process(self, min_silence_len=1000, silence_thresh=-40, keep_silence=100):
        """
        メイン処理を実行
        
        Args:
            min_silence_len (int): 無音と判定する最小長さ（ミリ秒）
            silence_thresh (int): 無音の閾値（dB）
            keep_silence (int): 分割後に残す無音の長さ（ミリ秒）
        """
        print("=" * 50)
        print("MP3ファイル分割処理を開始します")
        print("=" * 50)
        
        # 1. MP3ファイルを読み込み
        if not self.load_audio():
            return False
        
        # 2. 出力フォルダを作成
        if not self.create_output_folder():
            return False
        
        # 3. 無音箇所で音声を分割
        audio_chunks = self.split_audio(min_silence_len, silence_thresh, keep_silence)
        if not audio_chunks:
            print("分割された音声チャンクがありません")
            return False
        
        # 4. 分割された音声をWAVファイルとして保存
        saved_files = self.save_chunks(audio_chunks)
        
        print("=" * 50)
        print("処理完了!")
        print(f"分割されたファイル数: {len(saved_files)}")
        print(f"出力フォルダ: {self.output_folder}")
        print("=" * 50)
        
        return True

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法: python mp3_splitter.py <MP3ファイルパス> [無音判定長さ(ms)] [無音閾値(dB)] [無音保持長さ(ms)]")
        print("例: python mp3_splitter.py audio.mp3 1000 -40 100")
        return
    
    mp3_file_path = sys.argv[1]
    
    # オプション引数の設定
    min_silence_len = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    silence_thresh = int(sys.argv[3]) if len(sys.argv) > 3 else -40
    keep_silence = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    # ファイルの存在確認
    if not os.path.exists(mp3_file_path):
        print(f"エラー: ファイルが見つかりません: {mp3_file_path}")
        return
    
    # MP3Splitterインスタンスを作成して処理を実行
    splitter = MP3Splitter(mp3_file_path)
    success = splitter.process(min_silence_len, silence_thresh, keep_silence)
    
    if success:
        print("MP3ファイルの分割が正常に完了しました")
    else:
        print("MP3ファイルの分割に失敗しました")

if __name__ == "__main__":
    main()
