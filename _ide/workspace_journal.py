# 2026-10-24T10:11:19.369173642
import vitis

client = vitis.create_client()
client.set_workspace(path="HLS_llama")

proj = client.get_sys_project(name="llama_project")

status = proj.clean(target="hw")

proj.build(target = "hw",build_comps = False)

