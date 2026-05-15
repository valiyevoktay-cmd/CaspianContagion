import asyncio
import websockets

async def test():
    url = "wss://fstream.binance.com/ws/btcusdt@depth20@100ms"
    try:
        async with websockets.connect(url) as ws:
            print("✅ Connection Successful! Receiving data...")
            msg = await ws.recv()
            print(msg[:100])
    except Exception as e:
        print(f"❌ Error: {e}")

asyncio.run(test())