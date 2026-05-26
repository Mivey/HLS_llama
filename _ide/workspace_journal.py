# 2026-05-25T17:38:13.972298672
import vitis

client = vitis.create_client()
client.set_workspace(path="HLS_llama")

vitis.dispose()

