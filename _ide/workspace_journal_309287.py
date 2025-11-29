# 2025-11-28T11:34:06.184576954
import vitis

client = vitis.create_client()
client.set_workspace(path="forward")

platform = client.get_component(name="TE0950_plat")
status = platform.build()

comp = client.get_component(name="besty_test")
comp.build(target="hw")

status = platform.build()

comp.build(target="hw")

vitis.dispose()

