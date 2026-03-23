import os
import discord
from discord.ext import commands
import ntsc
import io
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def decode_image(image_buffer):
    image_array = np.frombuffer(image_buffer.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def process_image(image_buffer):
    src = decode_image(image_buffer)
    if src is None:
        raise ValueError("Failed to decode image")

    dst = src.copy()  # 出力先を分ける
    fx = ntsc.Ntsc()

    # 見えるように少し強めに設定
    fx._video_noise = 800
    fx._vhs_head_switching = True
    fx.freq_noise_size = 0.6

    fx._video_chroma_noise = 2000
    fx._ringing = 0.6
    fx._color_bleed_horiz = 2
    fx._color_bleed_vert = 1
    fx._emulating_vhs = True
    fx._vhs_edge_wave = 6

    fx.composite_layer(dst, src, field=0, fieldno=0)
    fx.composite_layer(dst, src, field=1, fieldno=1)

    out = io.BytesIO()
    out.write(cv2.imencode(".png", dst)[1].tobytes())  # まずPNGで確認
    out.seek(0)
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
    image_buffer = process_image(image_buffer)
    await ctx.send(file=discord.File(image_buffer, filename="processed_image.jpg"))

bot.run(TOKEN)
