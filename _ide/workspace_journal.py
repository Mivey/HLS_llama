# 2026-04-26T20:59:00.604158229
import vitis

client = vitis.create_client()
client.set_workspace(path="HLS_llama")

comp = client.get_component(name="transformer_cu")
comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

cfg = client.get_config_file(path="/home/lolwut/project/2026/HLS/HLS_llama/transformer_cu/hls_config.cfg")

cfg.set_value(section="hls", key="clock", value="3.2ns")

comp.run(operation="SYNTHESIS")

vitis.dispose()

