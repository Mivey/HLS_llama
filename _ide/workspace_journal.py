# 2025-11-28T16:57:33.698085936
import vitis

client = vitis.create_client()
client.set_workspace(path="forward")

proj = client.get_sys_project(name="system_project")

status = proj.clean(target="hw")

platform = client.get_component(name="TE0950_plat")
status = platform.build()

comp = client.get_component(name="besty_test")
comp.build(target="hw")

