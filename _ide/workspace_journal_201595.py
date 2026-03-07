# 2026-03-07T15:09:54.118864049
import vitis

client = vitis.create_client()
client.set_workspace(path="HLS_llama")

vitis.dispose()

