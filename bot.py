import os
import discord
from discord.ext import commands
import ntsc
import io
import numpy as np
import cv2
from dotenv import load_dotenv
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Thread pool for image processing
executor = ThreadPoolExecutor(max_workers=2)

def decode_image(image_buffer):
    image_array = np.frombuffer(image_buffer.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def process_image(image_buffer):
    start_time = time.time()
    src = decode_image(image_buffer)
    if src is None:
        raise ValueError("Failed to decode image")

    # 画像サイズのバリデーションと調整
    height, width = src.shape[:2]
    original_size = (height, width)
    
    # 最小サイズ チェック：小さすぎる場合はアップスケール
    MIN_SIZE = 480
    if height < MIN_SIZE or width < MIN_SIZE:
        scale = MIN_SIZE / min(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        src = cv2.resize(src, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        height, width = new_height, new_width
        print(f"[Image Upscaled] {original_size} -> {(height, width)}")
    
    # 最大サイズ チェック：大きすぎる場合はダウンスケール
    MAX_SIZE = 4096
    downscale_factor = 1.0
    if height > MAX_SIZE or width > MAX_SIZE:
        scale = MAX_SIZE / max(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        src = cv2.resize(src, (new_width, new_height), interpolation=cv2.INTER_AREA)
        height, width = new_height, new_width
        print(f"[Image Downscaled] {original_size} -> {(height, width)}")
    
    # 処理用にさらにダウンスケール（1024px 以上の場合は 50% スケーリング）
    if height >= 1024 or width >= 1024:
        downscale_factor = 0.5
        new_height, new_width = int(height * downscale_factor), int(width * downscale_factor)
        src = cv2.resize(src, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"[Image Processing Downscaled] {(height, width)} -> {(new_height, new_width)}")
        height, width = new_height, new_width

    dst = src.copy()  # 出力先を分ける
    fx = ntsc.Ntsc()

    # 画像サイズに応じた処理の最適化
    if height < 640 or width < 640:
        # 小さい画像の場合：FFT処理コスト削減のため ringing を無効化
        fx._ringing = 1.0  # No ringing effect
        fx._enable_ringing2 = False
        fx._vhs_edge_wave = 0  # Disable edge wave processing
        print("[Optimization] Small image detected: ringing & edge_wave disabled")
    else:
        # 通常設定
        fx._ringing = 0.6
        fx._enable_ringing2 = False
        fx._vhs_edge_wave = 6

    # 見えるように少し強めに設定
    fx._video_noise = 800
    fx._vhs_head_switching = True
    fx.freq_noise_size = 0.6

    fx._video_chroma_noise = 2000
    fx._color_bleed_horiz = 2
    fx._color_bleed_vert = 1
    fx._emulating_vhs = True

    fx.composite_layer(dst, src, field=0, fieldno=0)
    fx.composite_layer(dst, src, field=1, fieldno=1)

    # ダウンスケールした場合はアップスケール復元
    if downscale_factor < 1.0:
        new_height, new_width = int(height / downscale_factor), int(width / downscale_factor)
        dst = cv2.resize(dst, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"[Image Upscaled Back] {(height, width)} -> {(new_height, new_width)}")

    out = io.BytesIO()
    out.write(cv2.imencode(".png", dst)[1].tobytes())  # まずPNGで確認
    out.seek(0)
    
    elapsed_time = time.time() - start_time
    print(f"[Processing Complete] Elapsed time: {elapsed_time:.2f}s")
    
    return out

@bot.event
async def on_ready():
    print(f"logged in as {bot.user}")

@bot.command()
async def capture(ctx: commands.Context):
    if not ctx.message.attachments:
        await ctx.send("Please attach an image.")
        return
    
    attachment = ctx.message.attachments[0]

    if attachment.content_type is not None and not attachment.content_type.startswith("image/"):
        await ctx.send("Please attach a valid image.")
        return
    
    image_bytes = await attachment.read()
    image_buffer = io.BytesIO(image_bytes)
    image_buffer.name = attachment.filename

    await ctx.send("Processing image...")
    
    # Run image processing in thread pool to prevent blocking Discord heartbeat
    loop = asyncio.get_event_loop()
    try:
        image_buffer = await loop.run_in_executor(executor, process_image, image_buffer)
        await ctx.send(file=discord.File(image_buffer, filename="processed_image.png"))
    except Exception as e:
        print(f"[Error] Processing failed: {e}")
        await ctx.send(f"Error processing image: {str(e)}")

bot.run(TOKEN)
