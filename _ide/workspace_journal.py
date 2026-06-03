# 2026-06-02T22:22:09.097254781
import vitis

client = vitis.create_client()
client.set_workspace(path="HLS_llama")

comp = client.get_component(name="mha_cu")
comp.run(operation="SYNTHESIS")

cfg = client.get_config_file(path="/home/lolwut/project/2026/HLS/HLS_llama/mha_cu/mha_config.cfg")

cfg.set_value(section="hls", key="syn.top", value="old_mha_kernel")

comp.run(operation="SYNTHESIS")

cfg.set_value(section="hls", key="syn.top", value="mha_kernel")

comp.run(operation="SYNTHESIS")

vitis.dispose()

