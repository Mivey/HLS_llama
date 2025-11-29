# 2025-11-27T21:28:16.617163380
import vitis

client = vitis.create_client()
client.set_workspace(path="forward")

cfg = client.get_config_file(path="/home/lolwut/proj/2025/plat/forward/accel_llama/accelerated_llama/dataflow_forward/hls_config.cfg")

cfg.set_value(section="hls", key="package.output.format", value="xo")

cfg.set_value(section="hls", key="package.output.syn", value="1")

comp = client.get_component(name="dataflow_forward")
comp.run(operation="SYNTHESIS")

cfg.set_values(key="syn.file", values=["./mha.cpp", "./final_third.cpp", "./first_kernel.cpp", "./forward.cpp", "bottom.cpp", "./top.cpp"])

comp.run(operation="SYNTHESIS")

cfg.set_value(section="hls", key="syn.csimflags", value="")

cfg.set_value(section="hls", key="syn.cflags", value="")

cfg.set_values(key="syn.file_csimflags", values=[])

cfg.set_values(key="syn.file_cflags", values=[])

cfg.set_values(key="syn.file", values=["./mha.cpp", "./final_third.cpp", "./first_kernel.cpp", "./forward.cpp", "./top.cpp"])

comp.run(operation="SYNTHESIS")

cfg = client.get_config_file(path="/home/lolwut/proj/2025/plat/forward/system_project/hw_link/data_f-link.cfg")

cfg.add_lines(values=["sp=top_1.output_tokens:DDR"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.output_tokens:"])

cfg.add_lines(values=["chipscope=top_1:output_tokens"])

cfg.remove(section="debug", keysOrLines=["chipscope=top_1:output_tokens"])

cfg.add_lines(values=["sp=top_1.output_tokens:DDR"])

cfg.add_lines(values=["sp=top_1.key_cache_out:DDR"])

cfg.add_lines(values=["sp=top_1.key_cache:DDR"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.key_cache:"])

cfg.add_lines(values=["sp=top_1.key_cache:out:DDR"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.key_cache_out:"])

cfg.set_value(section="connectivity", key="sp=top_1.key_cache:", value="DDR")

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.key_cache:"])

cfg.add_lines(values=["sp=top_1.key_cache:DDR"])

cfg.add_lines(values=["sp=top_1.value_cache_out:DDR"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.value_cache_out:"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.output_tokens:"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.key_cache:"])

cfg.add_lines(values=["sp=top_1.swiglu_comp_weights_sf:DDR"])

cfg.add_lines(values=["sp=top_1.mlp_exp_weights3_sf:DDR"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.mlp_exp_weights3_sf:"])

cfg.remove(section="connectivity", keysOrLines=["sp=top_1.swiglu_comp_weights_sf:"])

platform = client.get_component(name="TE0950_plat")
status = platform.build()

comp = client.get_component(name="test_app")
comp.build(target="hw")

status = platform.build()

comp.build(target="hw")

comp = client.create_app_component(name="besty_test",platform = "$COMPONENT_LOCATION/../TE0950_plat/export/TE0950_plat/TE0950_plat.xpfm",domain = "linux_psv_cortexa72")

comp = client.get_component("besty_test")

status = comp.set_sysroot(sysroot="/home/lolwut/proj/2025/plat/forward/xilinx-versal-common-v2025.2/sysroots/cortexa72-cortexa53-amd-linux")

comp = client.get_component(name="besty_test")
status = comp.import_files(from_loc="", files=["/home/lolwut/proj/2025/tutorial/vadd_tutorial/vadd_host/src"], is_skip_copy_sources = False)

status = platform.build()

comp.build(target="hw")

status = platform.build()

comp = client.get_component(name="test_app")
comp.build(target="hw")

status = platform.build()

comp = client.get_component(name="besty_test")
comp.build(target="hw")

status = platform.build()

comp.build(target="hw")

status = platform.build()

comp = client.get_component(name="test_app")
comp.build(target="hw")

comp.set_app_config(key = "USER_COMPILE_SOURCES", values = ["/home/lolwut/proj/2025/plat/forward/test_app/src/runq.c"])

status = platform.build()

comp.build(target="hw")

client.delete_component(name="test_app")

comp = client.create_app_component(name="app_component",platform = "$COMPONENT_LOCATION/../TE0950_plat/export/TE0950_plat/TE0950_plat.xpfm",domain = "linux_psv_cortexa72")

comp = client.get_component("app_component")

status = comp.set_sysroot(sysroot="/home/lolwut/proj/2025/plat/forward/xilinx-versal-common-v2025.2/sysroots/cortexa72-cortexa53-amd-linux")

comp = client.get_component(name="app_component")
status = comp.import_files(from_loc="", files=["/home/lolwut/Downloads/runq.c"], is_skip_copy_sources = False)

status = platform.build()

comp.build(target="hw")

comp.set_app_config(key = "USER_COMPILE_OTHER_FLAGS", values = ["-fopenmp"])

comp.set_app_config(key = "USER_CMAKE_CXX_STANDARD", values = ["17"])

comp.set_app_config(key = "USER_CMAKE_CXX_STANDARD", values = [""])

comp = client.get_component(name="besty_test")
comp.set_app_config(key = "USER_CMAKE_CXX_STANDARD", values = ["17"])

status = platform.build()

comp = client.get_component(name="app_component")
comp.build(target="hw")

comp.set_app_config(key = "USER_LINK_LIBRARIES", values = ["m"])

status = platform.build()

comp.build(target="hw")

comp.set_app_config(key = "USER_LINK_LIBRARIES", values = [""])

status = platform.build()

comp.build(target="hw")

status = platform.build()

comp.build(target="hw")

status = platform.build()

comp = client.get_component(name="besty_test")
comp.build(target="hw")

proj = client.get_sys_project(name="system_project")

proj.build(target = "hw",comp_name = ["accel_llama/accelerated_llama/dataflow_forward"],build_comps = False)

vitis.dispose()

vitis.dispose()

